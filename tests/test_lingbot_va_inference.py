from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot_inference.inference.policies.lingbot_va import LingBotVAAdapter
from lerobot_inference.inference.policies.registry import (
    _create_lingbot_va,
    _lingbot_va_policy_config_overrides,
)


class FakeProcessor:
    def __init__(self) -> None:
        self.call_count = 0
        self.reset_count = 0

    def __call__(self, value):
        self.call_count += 1
        return value

    def reset(self) -> None:
        self.reset_count += 1


class FakeLingBotVAPolicy:
    def __init__(self, *, with_state: bool = False) -> None:
        input_features = {
            "observation.images.image": SimpleNamespace(shape=(3, 128, 128)),
            "observation.images.image2": SimpleNamespace(shape=(3, 128, 128)),
        }
        if with_state:
            input_features["observation.state"] = SimpleNamespace(shape=(8,))
        self.config = SimpleNamespace(
            device="cpu",
            input_features=input_features,
            output_features={"action": SimpleNamespace(shape=(7,))},
            used_action_channel_ids=[0, 1, 2, 3, 4, 5, 6],
        )
        self.reset_count = 0
        self.select_batches: list[dict] = []

    def parameters(self):
        return iter(())

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, batch: dict) -> torch.Tensor:
        self.select_batches.append(batch)
        return torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])


def make_adapter(
    *,
    instruction: str = "Place the red cube on top of the green cube",
    policy: FakeLingBotVAPolicy | None = None,
    preprocessor: FakeProcessor | None = None,
    postprocessor: FakeProcessor | None = None,
) -> LingBotVAAdapter:
    return LingBotVAAdapter(
        policy or FakeLingBotVAPolicy(),
        preprocessor or FakeProcessor(),
        postprocessor or FakeProcessor(),
        instruction=instruction,
        expected_image_keys={"observation.images.image", "observation.images.image2"},
    )


def make_observation() -> dict[str, np.ndarray]:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image.setflags(write=False)  # Arrow-backed image views are commonly read-only.
    return {
        "observation.images.image": image,
        "observation.images.image2": image,
    }


EXPECTED_ACTION = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_lingbot_va_injects_instruction_as_task_into_batch() -> None:
    policy = FakeLingBotVAPolicy()
    adapter = make_adapter(policy=policy, instruction="Pick up the cup")

    adapter.generate_action(make_observation())

    assert policy.select_batches[0]["task"] == "Pick up the cup"


def test_lingbot_va_always_needs_observation() -> None:
    adapter = make_adapter()

    assert adapter.is_observation_needed() is True
    with pytest.raises(ValueError, match="requires an observation on every step"):
        adapter.generate_action({})
    np.testing.assert_array_equal(adapter.generate_action(make_observation()), EXPECTED_ACTION)
    # The policy owns its chunk and KV cache internally, so the next step still needs a frame.
    assert adapter.is_observation_needed() is True


def test_lingbot_va_generate_action_pipeline_order() -> None:
    policy = FakeLingBotVAPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = make_adapter(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor)

    np.testing.assert_array_equal(adapter.generate_action(make_observation()), EXPECTED_ACTION)

    assert preprocessor.call_count == 1
    assert postprocessor.call_count == 1
    assert len(policy.select_batches) == 1
    batch = policy.select_batches[0]
    assert batch["observation.images.image"].shape == (1, 3, 128, 128)
    assert batch["observation.images.image2"].shape == (1, 3, 128, 128)
    assert batch["task"] == "Place the red cube on top of the green cube"


def test_lingbot_va_validates_observation_keys() -> None:
    adapter = make_adapter()

    observation = make_observation()
    del observation["observation.images.image2"]
    with pytest.raises(KeyError, match="Missing camera observations"):
        adapter.generate_action(observation)


def test_lingbot_va_validates_image_format() -> None:
    adapter = make_adapter()

    observation = make_observation()
    observation["observation.images.image"] = np.full((128, 128, 3), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="must contain finite values in \\[0, 1\\]"):
        adapter.generate_action(observation)


def test_lingbot_va_reset_resets_policy_and_processors() -> None:
    policy = FakeLingBotVAPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = make_adapter(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor)

    adapter.reset()

    assert policy.reset_count == 1
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1


def test_lingbot_va_pause_and_stop_reset() -> None:
    policy = FakeLingBotVAPolicy()
    adapter = make_adapter(policy=policy)

    adapter.pause()
    adapter.stop()

    assert policy.reset_count == 2


def test_lingbot_va_instruction_setter_resets_policy() -> None:
    policy = FakeLingBotVAPolicy()
    adapter = make_adapter(policy=policy)

    adapter.instruction = "A different task"
    assert adapter.instruction == "A different task"
    assert policy.reset_count == 1  # new prompt must be re-encoded from scratch

    # Setting the same instruction is a no-op (no reset).
    adapter.instruction = "A different task"
    assert policy.reset_count == 1


def test_lingbot_va_validate_io_dimensions_mismatch() -> None:
    adapter = make_adapter()

    with pytest.raises(ValueError, match="action_joints=6"):
        adapter.validate_io_dimensions(99, 6)  # state skipped, action must still match


def test_lingbot_va_validate_io_dimensions_ok() -> None:
    make_adapter().validate_io_dimensions(99, 7)  # no state feature: state dim ignored


def test_lingbot_va_validate_io_dimensions_with_state() -> None:
    policy = FakeLingBotVAPolicy(with_state=True)
    adapter = make_adapter(policy=policy)

    adapter.validate_io_dimensions(8, 7)
    with pytest.raises(ValueError, match="state_joints=6"):
        adapter.validate_io_dimensions(6, 7)


def test_lingbot_va_policy_config_overrides() -> None:
    assert _lingbot_va_policy_config_overrides({}) == {}
    overrides = _lingbot_va_policy_config_overrides(
        {
            "camera_names": ["image", "image2"],
            "num_inference_steps": 20,
            "action_num_inference_steps": 50,
            "wan_pretrained_path": "/local/frozen",
            "text_encoder_device": "cuda:1",
        }
    )
    assert overrides["obs_cam_keys"] == [
        "observation.images.image",
        "observation.images.image2",
    ]
    assert overrides["num_inference_steps"] == 20
    assert overrides["action_num_inference_steps"] == 50
    assert overrides["wan_pretrained_path"] == "/local/frozen"
    assert overrides["text_encoder_device"] == "cuda:1"
    # 无相机配置时不注入 obs_cam_keys。
    assert "obs_cam_keys" not in _lingbot_va_policy_config_overrides(
        {"num_inference_steps": 8}
    )


def test_lingbot_va_factory_validates_runtime_dimensions(monkeypatch) -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.state_dim = self.action_dim = None

        def validate_io_dimensions(self, state_dim, action_dim) -> None:
            self.state_dim, self.action_dim = state_dim, action_dim

    fake = FakeAdapter()
    monkeypatch.setattr(
        "lerobot_inference.inference.policies.registry.LingBotVAAdapter",
        SimpleNamespace(from_pretrained=lambda *a, **k: fake),
    )

    _create_lingbot_va(
        {
            "type": "lingbot_va",
            "state_joint_count": 8,
            "action_joint_count": 7,
            "camera_names": ["image", "image2"],
            "instruction": "Pick it up",
        },
        "/unused",
    )

    assert fake.state_dim == 8
    assert fake.action_dim == 7


def _monkeypatched_bundle(
    monkeypatch,
    *,
    type="lingbot_va",
    obs_cam_keys=("observation.images.image", "observation.images.image2"),
):
    """Stub load_policy_bundle to return a fake bundle whose config drives validation."""
    policy = FakeLingBotVAPolicy()
    policy.config.type = type
    policy.config.obs_cam_keys = list(obs_cam_keys)

    def fake_load(path, *, device=None, policy_config_overrides=None):
        assert path == "/unused"
        return policy, FakeProcessor(), FakeProcessor(), policy.config

    monkeypatch.setattr(
        "lerobot_inference.inference.policies.lingbot_va.adapter.load_policy_bundle",
        fake_load,
    )
    return policy


def test_lingbot_va_from_pretrained_rejects_wrong_policy_type(monkeypatch) -> None:
    _monkeypatched_bundle(monkeypatch, type="act")
    with pytest.raises(ValueError, match="expects type lingbot_va, got 'act'"):
        LingBotVAAdapter.from_pretrained("/unused", instruction="Pick it up")


def test_lingbot_va_from_pretrained_validates_camera_keys(monkeypatch) -> None:
    _monkeypatched_bundle(monkeypatch, obs_cam_keys=("observation.images.image", "observation.images.image2"))
    with pytest.raises(ValueError, match="camera mapping does not match checkpoint"):
        LingBotVAAdapter.from_pretrained(
            "/unused",
            instruction="Pick it up",
            expected_image_keys={"observation.images.image", "observation.images.side"},
        )


def test_lingbot_va_from_pretrained_ok(monkeypatch) -> None:
    policy = _monkeypatched_bundle(monkeypatch)
    adapter = LingBotVAAdapter.from_pretrained(
        "/unused",
        instruction="Pick it up",
        expected_image_keys={"observation.images.image", "observation.images.image2"},
    )
    assert adapter.instruction == "Pick it up"
    assert adapter._policy is policy
    assert adapter._expected_image_keys == {"observation.images.image", "observation.images.image2"}
