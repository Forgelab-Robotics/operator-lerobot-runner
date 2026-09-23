from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from lerobot_inference.cli import _normalize_argv, main
from lerobot_inference.inference.config import PolicyNodeConfig
from lerobot_inference.inference.libero_eval import (
    _write_json_atomic,
    build_fastwam_observation,
    prepare_libero_runtime,
    quat_to_axis_angle,
    rotate_libero_image,
    run_single_episode,
    validate_libero_task,
    validate_libero_module_origin,
)


def make_libero_observation() -> dict:
    primary = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.zeros((8, 8, 3), dtype=np.uint8)
    primary[0, 0] = [1, 2, 3]
    wrist[0, 0] = [4, 5, 6]
    return {
        "pixels": {"image": primary, "image2": wrist},
        "robot_state": {
            "eef": {
                "pos": np.array([0.1, 0.2, 0.3], dtype=np.float64),
                "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            },
            "gripper": {"qpos": np.array([0.01, -0.01], dtype=np.float64)},
        },
    }


class FakePolicy:
    def __init__(self, *, queue_size: int = 2, invalid: bool = False) -> None:
        self.queue_size = queue_size
        self.remaining = 0
        self.invalid = invalid
        self.reset_count = 0
        self.observations: list[dict] = []

    def reset(self) -> None:
        self.reset_count += 1
        self.remaining = 0

    def stop(self) -> None:
        self.reset()

    def is_observation_needed(self) -> bool:
        return self.remaining == 0

    def generate_action(self, observation, alias_for_cameras=None):
        assert alias_for_cameras == ["image", "image2"]
        self.observations.append(observation)
        if observation:
            self.remaining = self.queue_size
        assert self.remaining > 0
        self.remaining -= 1
        action = np.arange(7, dtype=np.float32) / 10
        if self.invalid:
            action[-1] = np.nan
        return action


class FakeEnv:
    def __init__(self, *, success_step: int | None = None, fail_step: int | None = None) -> None:
        self.success_step = success_step
        self.fail_step = fail_step
        self.step_count = 0

    def reset(self, seed=None, **kwargs):
        self.step_count = 0
        return make_libero_observation(), {"is_success": False}

    def step(self, action):
        self.step_count += 1
        if self.fail_step == self.step_count:
            raise RuntimeError("simulator failed")
        success = self.success_step == self.step_count
        return make_libero_observation(), float(success), success, False, {"is_success": success}

    def close(self) -> None:
        pass


class FakeVideoWriter:
    instances: list["FakeVideoWriter"] = []

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_quaternion_and_libero_observation_conversion() -> None:
    np.testing.assert_array_equal(
        quat_to_axis_angle(np.array([0, 0, 0, 1], dtype=np.float32)),
        np.zeros(3, dtype=np.float32),
    )
    axis_angle = quat_to_axis_angle(
        np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
    )
    np.testing.assert_allclose(axis_angle, [0, 0, np.pi / 2], rtol=1e-5)

    source = make_libero_observation()
    converted = build_fastwam_observation(source)
    assert converted["observation.state"].shape == (8,)
    assert converted["observation.state"].dtype == np.float32
    assert set(converted) == {
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
    }
    np.testing.assert_array_equal(converted["observation.images.image"][-1, -1], [1, 2, 3])


def test_rotate_libero_image_validates_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="HWC RGB"):
        rotate_libero_image(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        rotate_libero_image(np.zeros((8, 8, 3), dtype=np.float32))


def test_prepare_runtime_and_validate_source(tmp_path: Path) -> None:
    root = tmp_path / "LIBERO"
    benchmark_root = root / "libero" / "libero"
    for path in (
        root / "libero" / "__init__.py",
        benchmark_root / "__init__.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    for directory in ("bddl_files", "init_files", "assets"):
        (benchmark_root / directory).mkdir()

    paths = prepare_libero_runtime(root, tmp_path / "run")
    runtime_config = yaml.safe_load(
        (tmp_path / "run" / "libero_config" / "config.yaml").read_text()
    )
    assert runtime_config["assets"] == str(benchmark_root / "assets")
    assert runtime_config["datasets"] == str(tmp_path / "run" / "libero_config" / "datasets")
    assert paths["libero_root"] == str(root.resolve())
    assert Path(sys.path[0]).resolve() == root.resolve()

    local_module = ModuleType("libero.libero")
    local_module.__file__ = str(benchmark_root / "__init__.py")
    assert validate_libero_module_origin(local_module, root) == (
        benchmark_root / "__init__.py"
    ).resolve()

    foreign_module = ModuleType("libero.libero")
    foreign_module.__file__ = str(tmp_path / "site-packages" / "libero" / "__init__.py")
    with pytest.raises(RuntimeError, match="did not come from"):
        validate_libero_module_origin(foreign_module, root)


def test_validate_libero_task() -> None:
    task = SimpleNamespace(language="do the task")
    suite = SimpleNamespace(tasks=[task], get_task=lambda task_id: [task][task_id])
    assert validate_libero_task(suite, "libero_spatial", 0) is task
    with pytest.raises(ValueError, match="outside suite"):
        validate_libero_task(suite, "libero_spatial", 1)
    with pytest.raises(ValueError, match="contains no tasks"):
        validate_libero_task(SimpleNamespace(tasks=[]), "empty", 0)


def test_run_single_episode_uses_action_queue_and_writes_success(tmp_path: Path) -> None:
    FakeVideoWriter.instances.clear()
    action_log = io.StringIO()
    policy = FakePolicy(queue_size=2)
    result = run_single_episode(
        env=FakeEnv(success_step=3),
        policy=policy,
        task_description="pick up the bowl",
        episode_index=0,
        init_state_id=0,
        seed=0,
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
    assert [bool(observation) for observation in policy.observations] == [True, False, True]
    rows = [json.loads(line) for line in action_log.getvalue().splitlines()]
    assert len(rows) == 3
    assert rows[0]["new_prediction"] is True
    assert rows[1]["new_prediction"] is False
    writer = FakeVideoWriter.instances[-1]
    assert writer.closed is True
    assert writer.fps == 20
    assert len(writer.frames) == 4
    assert writer.frames[0].shape == (8, 16, 3)


def test_run_single_episode_horizon_is_valid_failure(tmp_path: Path) -> None:
    result = run_single_episode(
        env=FakeEnv(),
        policy=FakePolicy(queue_size=2),
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
    assert result["success"] is False
    assert result["termination"] == "horizon"
    assert result["steps"] == 2


@pytest.mark.parametrize(
    ("env", "policy", "message"),
    [
        (FakeEnv(), FakePolicy(invalid=True), "finite 7D"),
        (FakeEnv(fail_step=1), FakePolicy(), "simulator failed"),
    ],
)
def test_episode_errors_still_close_video(
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


def test_result_json_is_atomic_and_unicode_safe(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    _write_json_atomic(path, {"status": "completed", "message": "成功"})
    assert json.loads(path.read_text()) == {"status": "completed", "message": "成功"}


def test_libero_example_config_and_cli() -> None:
    config = PolicyNodeConfig.from_yaml_path(
        Path("examples/fastwam_libero_eval/policy_fastwam.yaml")
    )
    assert config.policy["type"] == "fastwam"
    assert config.alias_for_cameras == ["image", "image2"]
    assert _normalize_argv(["eval-libero", "--help"]) == ["eval-libero", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main(["eval-libero", "--help"])
    assert exc_info.value.code == 0
