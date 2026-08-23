from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lerobot_inference.cli import _normalize_argv, main
from lerobot_inference.inference.config import PolicyNodeConfig
from lerobot_inference.inference.libero_eval import (
    build_libero_observation,
    quat_to_axis_angle,
    run_libero_eval,
    run_single_episode,
)


def _observation() -> dict[str, Any]:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]
    return {
        "pixels": {"image": image, "image2": image.copy()},
        "robot_state": {
            "eef": {
                "pos": np.array([0.1, 0.2, 0.3]),
                "quat": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            "gripper": {"qpos": np.array([0.01, -0.01])},
        },
    }


class FakePolicy:
    def __init__(self, *, queue_size: int = 2, invalid_action: bool = False) -> None:
        self.queue_size = queue_size
        self.invalid_action = invalid_action
        self.remaining_actions = 0
        self.reset_count = 0
        self.observations: list[dict[str, Any]] = []
        self.camera_aliases: list[list[str] | None] = []

    def reset(self) -> None:
        self.reset_count += 1
        self.remaining_actions = 0

    def is_observation_needed(self) -> bool:
        return self.remaining_actions == 0

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        self.observations.append(observation)
        self.camera_aliases.append(alias_for_cameras)
        if observation:
            self.remaining_actions = self.queue_size
        self.remaining_actions -= 1
        action = np.arange(7, dtype=np.float32) / 10
        if self.invalid_action:
            action[-1] = np.nan
        return action


class FakeEnv:
    def __init__(self, *, success_step: int | None = None, fail_step: int | None = None) -> None:
        self.success_step = success_step
        self.fail_step = fail_step
        self.step_count = 0
        self.reset_seeds: list[int | None] = []
        self.actions: list[np.ndarray] = []

    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.step_count = 0
        self.reset_seeds.append(seed)
        return _observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self.actions.append(action.copy())
        if self.fail_step == self.step_count:
            raise RuntimeError("simulator failed")
        success = self.success_step == self.step_count
        return _observation(), float(success), success, False, {"is_success": success}

    def close(self) -> None:
        pass


class FakeVideoWriter:
    instances: list[FakeVideoWriter] = []

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.closed = False
        self.__class__.instances.append(self)

    def append(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeVideoWriter:
        return self

    def __exit__(self, exc_type: Any, exc_value: BaseException | None, tb: Any) -> None:
        self.close()


def test_build_libero_observation_returns_eight_state_and_two_cameras() -> None:
    converted = build_libero_observation(_observation())

    assert converted["observation.state"].shape == (8,)
    assert converted["observation.state"].dtype == np.float32
    assert set(converted) == {
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
    }
    np.testing.assert_array_equal(converted["observation.images.image"][-1, -1], [1, 2, 3])


def test_build_libero_observation_rejects_invalid_state() -> None:
    observation = _observation()
    observation["robot_state"]["gripper"]["qpos"] = np.zeros(1)

    with pytest.raises(ValueError, match="8D"):
        build_libero_observation(observation)


def test_quat_to_axis_angle_preserves_known_rotation() -> None:
    angle = np.pi / 2
    quaternion = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])

    np.testing.assert_allclose(
        quat_to_axis_angle(quaternion),
        [0.0, 0.0, angle],
        rtol=1e-5,
        atol=1e-6,
    )


def test_run_single_episode_closes_the_action_queue_loop(tmp_path: Path) -> None:
    FakeVideoWriter.instances.clear()
    policy = FakePolicy(queue_size=2)
    env = FakeEnv(success_step=3)
    action_log = io.StringIO()

    result = run_single_episode(
        env=env,
        policy=policy,
        task_description="pick up the bowl",
        episode_index=0,
        init_state_id=4,
        seed=9,
        episode_length=5,
        control_freq=20,
        video_path=tmp_path / "rollout.mp4",
        action_log=action_log,
        video_writer_factory=FakeVideoWriter,
    )

    assert result["success"] is True
    assert result["termination"] == "success"
    assert result["steps"] == 3
    assert result["replans"] == 2
    assert policy.reset_count == 1
    assert env.reset_seeds == [9]
    assert [bool(observation) for observation in policy.observations] == [True, False, True]
    assert set(policy.observations[0]) == {
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
    }
    assert policy.observations[1] == {}
    assert policy.camera_aliases == [["image", "image2"]] * 3
    assert len(env.actions) == 3
    assert all(action.shape == (7,) and np.isfinite(action).all() for action in env.actions)

    rows = [json.loads(line) for line in action_log.getvalue().splitlines()]
    assert [row["new_prediction"] for row in rows] == [True, False, True]
    assert [row["init_state_id"] for row in rows] == [4, 4, 4]

    writer = FakeVideoWriter.instances[-1]
    assert writer.closed is True
    assert writer.fps == 20
    assert len(writer.frames) == 4
    assert writer.frames[0].shape == (4, 10, 3)


@pytest.mark.parametrize(
    ("env", "policy", "message"),
    [
        (FakeEnv(), FakePolicy(invalid_action=True), "finite 7D"),
        (FakeEnv(fail_step=1), FakePolicy(), "simulator failed"),
    ],
)
def test_run_single_episode_closes_video_on_error(
    tmp_path: Path,
    env: FakeEnv,
    policy: FakePolicy,
    message: str,
) -> None:
    FakeVideoWriter.instances.clear()

    with pytest.raises(Exception, match=message):
        run_single_episode(
            env=env,
            policy=policy,
            task_description="pick up the bowl",
            episode_index=0,
            init_state_id=0,
            seed=0,
            episode_length=2,
            control_freq=20,
            video_path=tmp_path / "rollout.mp4",
            action_log=io.StringIO(),
            video_writer_factory=FakeVideoWriter,
        )

    assert FakeVideoWriter.instances[-1].closed is True


def test_eval_libero_public_handler_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    import lerobot_inference.inference.libero_eval as libero_eval

    received: list[Any] = []

    def fake_run(args: Any) -> int:
        received.append(args)
        return 17

    monkeypatch.setattr(libero_eval, "run_libero_eval", fake_run)
    result = main(
        [
            "eval-libero",
            "--config",
            "policy.yaml",
            "--libero-root",
            "/tmp/LIBERO",
            "--suite",
            "libero_spatial",
            "--task-id",
            "3",
        ]
    )

    assert callable(run_libero_eval)
    assert result == 17
    assert len(received) == 1
    assert received[0].config == "policy.yaml"
    assert received[0].libero_root == "/tmp/LIBERO"
    assert received[0].task_id == 3


def test_eval_run_records_initialization_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lerobot_inference.inference.libero_eval as libero_eval

    old_result = tmp_path / "20000101_000000" / "result.json"
    old_result.parent.mkdir()
    old_result.write_text('{"status": "completed"}\n', encoding="utf-8")

    def fail_runtime(libero_root: str | Path, run_dir: Path) -> dict[str, str]:
        raise RuntimeError("runtime initialization failed")

    monkeypatch.setattr(libero_eval, "prepare_libero_runtime", fail_runtime)
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        episodes=1,
        init_state_id=0,
        episode_length=10,
        num_steps_wait=0,
        control_freq=20,
        observation_size=64,
        libero_root="/tmp/LIBERO",
    )

    with pytest.raises(RuntimeError, match="runtime initialization failed"):
        run_libero_eval(args)

    assert json.loads(old_result.read_text(encoding="utf-8")) == {"status": "completed"}
    new_results = [path for path in tmp_path.glob("*/result.json") if path != old_result]
    assert len(new_results) == 1
    result = json.loads(new_results[0].read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert result["error"]["type"] == "RuntimeError"
    assert result["error"]["message"] == "runtime initialization failed"
    assert "fail_runtime" in result["error"]["traceback"]
    assert result["duration_s"] >= 0
    assert result["end_time"]
    assert Path(result["artifacts"]["actions"]).is_file()
    assert Path(result["artifacts"]["log"]).is_file()


def test_eval_libero_help_and_vla_config() -> None:
    assert _normalize_argv(["eval-libero", "--help"]) == ["eval-libero", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main(["eval-libero", "--help"])
    assert exc_info.value.code == 0

    config_path = Path(__file__).parents[1] / "examples" / "libero_eval" / "policy_vla_jepa.yaml"
    config = PolicyNodeConfig.from_yaml_path(config_path)
    assert config.policy["type"] == "vla_jepa"
    assert config.state_joint_order == [
        "eef_x",
        "eef_y",
        "eef_z",
        "axis_angle_x",
        "axis_angle_y",
        "axis_angle_z",
        "gripper_left",
        "gripper_right",
    ]
    assert config.alias_for_cameras == ["image", "image2"]
