from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot_inference.inference.policies.diffusion import DiffusionPolicyAdapter
from lerobot_inference.inference.policies.registry import (
    _create_diffusion,
    _diffusion_checkpoint_contract,
    _diffusion_instruction_conditioning,
    _diffusion_policy_config_overrides,
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


class FakeDiffusionPolicy:
    def __init__(self, *, with_state: bool = True) -> None:
        input_features = {
            "observation.images.top": SimpleNamespace(shape=(3, 224, 224)),
        }
        if with_state:
            input_features["observation.state"] = SimpleNamespace(shape=(8,))
        self.config = SimpleNamespace(
            device="cpu",
            use_amp=False,
            input_features=input_features,
            output_features={"action": SimpleNamespace(shape=(7,))},
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
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
    policy: FakeDiffusionPolicy | None = None,
    preprocessor: FakeProcessor | None = None,
    postprocessor: FakeProcessor | None = None,
) -> DiffusionPolicyAdapter:
    return DiffusionPolicyAdapter(
        policy or FakeDiffusionPolicy(),
        preprocessor or FakeProcessor(),
        postprocessor or FakeProcessor(),
        expected_image_keys={"observation.images.top"},
    )


def make_observation() -> dict[str, np.ndarray]:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image.setflags(write=False)  # Arrow-backed image views are commonly read-only.
    return {
        "observation.state": np.array([0.1] * 8, dtype=np.float64),
        "observation.images.top": image,
    }


EXPECTED_ACTION = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_diffusion_always_needs_observation() -> None:
    adapter = make_adapter()

    assert adapter.is_observation_needed() is True
    with pytest.raises(ValueError, match="requires an observation on every step"):
        adapter.generate_action({})
    np.testing.assert_array_equal(adapter.generate_action(make_observation()), EXPECTED_ACTION)
    # The policy owns its action chunk internally, so the next step still needs a frame.
    assert adapter.is_observation_needed() is True


def test_diffusion_generate_action_pipeline_order() -> None:
    policy = FakeDiffusionPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = make_adapter(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor)

    np.testing.assert_array_equal(adapter.generate_action(make_observation()), EXPECTED_ACTION)

    assert preprocessor.call_count == 1
    assert postprocessor.call_count == 1
    assert len(policy.select_batches) == 1
    batch = policy.select_batches[0]
    assert batch["observation.state"].shape == (1, 8)
    assert batch["observation.state"].dtype == torch.float32
    assert batch["observation.images.top"].shape == (1, 3, 224, 224)


def test_diffusion_validates_checkpoint_image_shape() -> None:
    adapter = DiffusionPolicyAdapter(
        FakeDiffusionPolicy(),
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.top"},
        expected_image_shapes={"observation.images.top": (3, 224, 224)},
    )
    observation = make_observation()
    observation["observation.images.top"] = np.zeros((256, 256, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not match checkpoint"):
        adapter.generate_action(observation)


def test_diffusion_validates_observation_keys() -> None:
    adapter = make_adapter()

    observation = make_observation()
    del observation["observation.images.top"]
    with pytest.raises(KeyError, match="Missing camera observations"):
        adapter.generate_action(observation)


def test_diffusion_validates_observation_state() -> None:
    adapter = make_adapter()

    observation = make_observation()
    del observation["observation.state"]
    with pytest.raises(KeyError, match="Missing observation.state"):
        adapter.generate_action(observation)

    observation = make_observation()
    observation["observation.state"] = np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    with pytest.raises(ValueError, match="finite"):
        adapter.generate_action(observation)


def test_diffusion_skips_state_check_for_pure_visual_config() -> None:
    policy = FakeDiffusionPolicy(with_state=False)
    adapter = make_adapter(policy=policy)

    observation = make_observation()
    del observation["observation.state"]
    np.testing.assert_array_equal(adapter.generate_action(observation), EXPECTED_ACTION)


def test_diffusion_reset_resets_policy_and_processors() -> None:
    policy = FakeDiffusionPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = make_adapter(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor)

    adapter.reset()

    assert policy.reset_count == 1
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1


def test_diffusion_instruction_is_explicitly_language_free() -> None:
    adapter = make_adapter()
    adapter.instruction = "put the bowl on the plate"

    assert adapter.instruction == "put the bowl on the plate"
    assert _diffusion_instruction_conditioning({}) == "none"
    assert _diffusion_instruction_conditioning(
        {"instruction_conditioning": "NONE"}
    ) == "none"
    with pytest.raises(ValueError, match="must be 'none'"):
        _diffusion_instruction_conditioning(
            {"instruction_conditioning": "distilbert"}
        )
    with pytest.raises(ValueError, match="instruction_conditioning='none'"):
        DiffusionPolicyAdapter(
            FakeDiffusionPolicy(),
            FakeProcessor(),
            FakeProcessor(),
            expected_image_keys={"observation.images.top"},
            instruction_conditioning="distilbert",
        )


def test_diffusion_pause_and_stop_reset() -> None:
    policy = FakeDiffusionPolicy()
    adapter = make_adapter(policy=policy)

    adapter.pause()
    adapter.stop()

    assert policy.reset_count == 2


def test_diffusion_validate_io_dimensions_mismatch() -> None:
    adapter = make_adapter()

    with pytest.raises(ValueError, match="action_joints=6"):
        adapter.validate_io_dimensions(8, 6)
    with pytest.raises(ValueError, match="state_joints=6"):
        adapter.validate_io_dimensions(6, 7)


def test_diffusion_validate_io_dimensions_ok() -> None:
    make_adapter().validate_io_dimensions(8, 7)


def test_diffusion_validate_io_dimensions_skips_state_for_pure_visual() -> None:
    policy = FakeDiffusionPolicy(with_state=False)
    make_adapter(policy=policy).validate_io_dimensions(99, 7)


def test_diffusion_policy_config_overrides_passthrough() -> None:
    assert _diffusion_policy_config_overrides({}) == {}
    assert _diffusion_policy_config_overrides(
        {"num_inference_steps": 10, "n_action_steps": 32, "compile_model": True}
    ) == {"num_inference_steps": 10, "n_action_steps": 32, "compile_model": True}
    # 未提供的运行键不注入默认值。
    assert _diffusion_policy_config_overrides({"compile_mode": "max-autotune"}) == {
        "compile_mode": "max-autotune"
    }


def test_diffusion_checkpoint_contract_from_runtime_config() -> None:
    assert _diffusion_checkpoint_contract(
        {
            "expected_n_obs_steps": 2,
            "expected_horizon": "16",
            "expected_n_action_steps": 8,
        }
    ) == {"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8}
    for invalid in (True, 0, 1.5, "not-an-int"):
        with pytest.raises(ValueError, match="positive integer"):
            _diffusion_checkpoint_contract({"expected_horizon": invalid})


def test_diffusion_factory_validates_runtime_dimensions(monkeypatch) -> None:
    class FakeAdapter:
        def __init__(self) -> None:
            self.state_dim = self.action_dim = None

        def validate_io_dimensions(self, state_dim, action_dim) -> None:
            self.state_dim, self.action_dim = state_dim, action_dim

    fake = FakeAdapter()
    monkeypatch.setattr(
        "lerobot_inference.inference.policies.registry.DiffusionPolicyAdapter",
        SimpleNamespace(from_pretrained=lambda *a, **k: fake),
    )

    _create_diffusion(
        {
            "type": "diffusion",
            "state_joint_count": 8,
            "action_joint_count": 7,
            "camera_names": ["top"],
        },
        "/unused",
    )

    assert fake.state_dim == 8
    assert fake.action_dim == 7


def test_diffusion_factory_passes_explicit_expected_image_keys(monkeypatch) -> None:
    """历史 checkpoint 的非标准图像键（如 pusht 的 observation.image）可显式声明。"""
    seen: dict = {}

    class FakeAdapter:
        def __init__(self) -> None:
            self.state_dim = self.action_dim = None

        def validate_io_dimensions(self, state_dim, action_dim) -> None:
            self.state_dim, self.action_dim = state_dim, action_dim

    fake = FakeAdapter()
    monkeypatch.setattr(
        "lerobot_inference.inference.policies.registry.DiffusionPolicyAdapter",
        SimpleNamespace(from_pretrained=lambda *a, **k: seen.update(k) or fake),
    )

    _create_diffusion(
        {
            "type": "diffusion",
            "state_joint_count": 2,
            "action_joint_count": 2,
            "expected_image_keys": ["observation.image"],
        },
        "/unused",
    )

    assert seen["expected_image_keys"] == {"observation.image"}
    assert fake.state_dim == 2
    assert fake.action_dim == 2


def _monkeypatched_bundle(monkeypatch, *, type="diffusion", image_features=("observation.images.top",), with_state=True):
    """Stub load_policy_bundle to return a fake bundle whose config drives validation."""
    policy = FakeDiffusionPolicy(with_state=with_state)
    policy.config.type = type
    policy.config.image_features = list(image_features)

    def fake_load(path, *, device=None, policy_config_overrides=None):
        assert path == "/unused"
        return policy, FakeProcessor(), FakeProcessor(), policy.config

    monkeypatch.setattr(
        "lerobot_inference.inference.policies.diffusion.adapter.load_policy_bundle",
        fake_load,
    )
    return policy


def test_diffusion_from_pretrained_rejects_wrong_policy_type(monkeypatch) -> None:
    _monkeypatched_bundle(monkeypatch, type="act")
    with pytest.raises(ValueError, match="expects type diffusion, got 'act'"):
        DiffusionPolicyAdapter.from_pretrained("/unused")


def test_diffusion_from_pretrained_validates_camera_keys(monkeypatch) -> None:
    _monkeypatched_bundle(monkeypatch, image_features=("observation.images.top",))
    with pytest.raises(ValueError, match="camera mapping does not match checkpoint"):
        DiffusionPolicyAdapter.from_pretrained(
            "/unused",
            expected_image_keys={"observation.images.top", "observation.images.side"},
        )


def test_diffusion_from_pretrained_ok(monkeypatch) -> None:
    policy = _monkeypatched_bundle(monkeypatch)
    adapter = DiffusionPolicyAdapter.from_pretrained(
        "/unused",
        expected_image_keys={"observation.images.top"},
    )
    assert adapter._policy is policy
    assert adapter._expected_image_keys == {"observation.images.top"}
    assert adapter._expected_image_shapes == {"observation.images.top": (3, 224, 224)}


def test_diffusion_from_pretrained_uses_checkpoint_temporal_contract(monkeypatch) -> None:
    policy = _monkeypatched_bundle(monkeypatch)
    policy.config.n_obs_steps = 3
    policy.config.horizon = 20
    policy.config.n_action_steps = 6

    DiffusionPolicyAdapter.from_pretrained(
        "/unused",
        expected_checkpoint_contract={
            "n_obs_steps": 3,
            "horizon": 20,
            "n_action_steps": 6,
        },
    )


def test_diffusion_from_pretrained_rejects_temporal_mismatch(monkeypatch) -> None:
    _monkeypatched_bundle(monkeypatch)
    with pytest.raises(
        ValueError, match="n_action_steps: expected=4, checkpoint=8"
    ):
        DiffusionPolicyAdapter.from_pretrained(
            "/unused",
            expected_checkpoint_contract={"n_obs_steps": 2, "n_action_steps": 4},
        )


def test_diffusion_from_pretrained_rejects_invalid_checkpoint_contract(monkeypatch) -> None:
    policy = _monkeypatched_bundle(monkeypatch)
    policy.config.horizon = 0

    with pytest.raises(ValueError, match="checkpoint has invalid horizon"):
        DiffusionPolicyAdapter.from_pretrained("/unused")
