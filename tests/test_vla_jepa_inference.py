from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot_inference.inference.policies.vla_jepa import VLAJEPAAdapter
from lerobot_inference.inference.policies.registry import (
    _create_vla_jepa,
    _vla_jepa_policy_config_overrides,
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


class FailOnceProcessor(FakeProcessor):
    def __call__(self, value):
        super().__call__(value)
        if self.call_count == 1:
            raise RuntimeError("postprocessor failed")
        return value


class FakeVLAJEPAPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            device="cpu",
            n_action_steps=7,
            input_features={
                "observation.state": SimpleNamespace(shape=(8,)),
                "observation.images.top": SimpleNamespace(shape=(3, 224, 224)),
                "observation.images.front": SimpleNamespace(shape=(3, 224, 224)),
            },
            output_features={"action": SimpleNamespace(shape=(7,))},
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
    policy: FakeVLAJEPAPolicy | None = None,
    preprocessor: FakeProcessor | None = None,
    postprocessor: FakeProcessor | None = None,
) -> VLAJEPAAdapter:
    return VLAJEPAAdapter(
        policy or FakeVLAJEPAPolicy(),
        preprocessor or FakeProcessor(),
        postprocessor or FakeProcessor(),
        instruction=instruction,
        expected_image_keys={"observation.images.top", "observation.images.front"},
    )


def make_observation() -> dict[str, np.ndarray]:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image.setflags(write=False)  # Arrow-backed image views are commonly read-only.
    return {
        "observation.state": np.array([0.1] * 8, dtype=np.float64),
        "observation.images.top": image,
        "observation.images.front": image,
    }


EXPECTED_ACTION = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_vla_jepa_injects_instruction_as_task_into_batch() -> None:
    policy = FakeVLAJEPAPolicy()
    adapter = make_adapter(policy=policy, instruction="Pick up the cup")

    adapter.generate_action(make_observation())

    assert policy.select_batches[0]["task"] == "Pick up the cup"


def test_vla_jepa_preprocesses_once_per_action_chunk() -> None:
    policy = FakeVLAJEPAPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = make_adapter(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor)

    assert adapter.is_observation_needed() is True
    np.testing.assert_array_equal(adapter.generate_action(make_observation()), EXPECTED_ACTION)
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), EXPECTED_ACTION)
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), EXPECTED_ACTION)
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), EXPECTED_ACTION)
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), EXPECTED_ACTION)
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), EXPECTED_ACTION)
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), EXPECTED_ACTION)
    assert adapter.is_observation_needed() is True  # 7-step chunk fully consumed

    assert preprocessor.call_count == 1
    assert postprocessor.call_count == 7
    assert len(policy.select_batches) == 7
    assert policy.select_batches[1:] == [{}, {}, {}, {}, {}, {}]
    assert policy.select_batches[0]["observation.state"].shape == (1, 8)
    assert policy.select_batches[0]["observation.state"].dtype == torch.float32
    assert policy.select_batches[0]["observation.images.top"].shape == (1, 3, 224, 224)


def test_vla_jepa_rejects_observation_while_chunk_is_queued() -> None:
    adapter = make_adapter()

    adapter.generate_action(make_observation())
    with pytest.raises(ValueError, match="still has queued actions"):
        adapter.generate_action(make_observation())


def test_vla_jepa_queue_stays_synchronized_when_postprocessing_fails() -> None:
    policy = FakeVLAJEPAPolicy()
    postprocessor = FailOnceProcessor()
    adapter = make_adapter(policy=policy, postprocessor=postprocessor)

    with pytest.raises(RuntimeError, match="postprocessor failed"):
        adapter.generate_action(make_observation())

    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), EXPECTED_ACTION)
    assert policy.select_batches[-1] == {}


def test_vla_jepa_reset_resets_policy_and_processors() -> None:
    policy = FakeVLAJEPAPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = make_adapter(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor)

    adapter.reset()

    assert policy.reset_count == 1
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1


def test_vla_jepa_instruction_setter_clears_queue() -> None:
    policy = FakeVLAJEPAPolicy()
    adapter = make_adapter(policy=policy)

    adapter.generate_action(make_observation())
    assert adapter.is_observation_needed() is False

    adapter.instruction = "A different task"
    assert adapter.instruction == "A different task"
    assert adapter.is_observation_needed() is True  # queue invalidated
    assert policy.reset_count == 1

    # Setting the same instruction is a no-op (no reset).
    adapter.instruction = "A different task"
    assert policy.reset_count == 1


def test_vla_jepa_rejects_observation_after_reset_without_fresh_obs() -> None:
    adapter = make_adapter()

    adapter.generate_action(make_observation())
    adapter.reset()
    with pytest.raises(ValueError, match="fresh observation is required"):
        adapter.generate_action({})


def test_vla_jepa_validate_io_dimensions_mismatch() -> None:
    adapter = make_adapter()

    with pytest.raises(ValueError, match="state_joints=6"):
        adapter.validate_io_dimensions(6, 7)
    with pytest.raises(ValueError, match="action_joints=6"):
        adapter.validate_io_dimensions(8, 6)


def test_vla_jepa_validate_io_dimensions_ok() -> None:
    make_adapter().validate_io_dimensions(8, 7)


def test_vla_jepa_policy_config_overrides_passthrough() -> None:
    assert _vla_jepa_policy_config_overrides({}) == {}
    assert _vla_jepa_policy_config_overrides(
        {"num_inference_timesteps": 8, "enable_world_model": False}
    ) == {"num_inference_timesteps": 8, "enable_world_model": False}
    # 未提供的运行键不注入默认值。
    assert _vla_jepa_policy_config_overrides({"qwen_model_name": "local/qwen"}) == {
        "qwen_model_name": "local/qwen"
    }


def test_vla_jepa_factory_validates_runtime_dimensions(monkeypatch) -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.state_dim = self.action_dim = None

        def validate_io_dimensions(self, state_dim, action_dim) -> None:
            self.state_dim, self.action_dim = state_dim, action_dim

    fake = FakeAdapter()
    monkeypatch.setattr(
        "lerobot_inference.inference.policies.registry.VLAJEPAAdapter",
        SimpleNamespace(from_pretrained=lambda *a, **k: fake),
    )

    _create_vla_jepa(
        {
            "type": "vla_jepa",
            "state_joint_count": 8,
            "action_joint_count": 7,
            "camera_names": ["top", "front"],
            "instruction": "Pick it up",
        },
        "/unused",
    )

    assert fake.state_dim == 8
    assert fake.action_dim == 7
