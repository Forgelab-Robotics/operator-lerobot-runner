from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot_inference.inference.config import JointConfig, PolicyNodeConfig
from lerobot_inference.inference.main import _build_joint_command
from lerobot_inference.inference.policies import loader
from lerobot_inference.inference.policies.act import ACTPolicyAdapter
from lerobot_inference.inference.policies.act.adapter import _select_act_image_features
from lerobot_inference.inference.policies.registry import (
    _act_policy_config_overrides,
    _act_runtime_dimensions,
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


class FakeACTPolicy:
    def __init__(self, *, temporal_ensemble_coeff: float | None = None) -> None:
        self.config = SimpleNamespace(
            device="cpu",
            use_amp=False,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            n_action_steps=1 if temporal_ensemble_coeff is not None else 3,
            input_features={
                "observation.state": SimpleNamespace(shape=(3,)),
                "observation.images.front": SimpleNamespace(shape=(3, 4, 5)),
            },
            output_features={"action": SimpleNamespace(shape=(2,))},
        )
        self.reset_count = 0
        self.select_batches: list[dict] = []
        # A non-empty private queue must not influence the adapter contract.
        self._action_queue = [torch.tensor([[99.0, 99.0]])]

    def parameters(self):
        return iter(())

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, batch: dict) -> torch.Tensor:
        self.select_batches.append(batch)
        return torch.tensor([[1.0, 2.0]])


def make_config(*, separate_state: bool = False) -> PolicyNodeConfig:
    data = {
        "joints": [
            {"name": "joint1", "mode": "position"},
            {"name": "joint2", "mode": "velocity"},
        ],
        "policy": {"type": "ACT", "pretrained_path": "/unused"},
        "image_inputs": {"image/front": "front"},
    }
    if separate_state:
        data["state_joints"] = ["joint1", "joint2", "sensor_joint"]
    return PolicyNodeConfig.from_dict(data)


def make_observation() -> dict[str, np.ndarray]:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image.setflags(write=False)  # Arrow-backed image views are commonly read-only.
    return {
        # JointState.to_np() returns float64 on the online Dora path.
        "observation.state": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "observation.images.front": image,
    }


def test_standard_act_preprocesses_once_per_action_chunk() -> None:
    policy = FakeACTPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = ACTPolicyAdapter(
        policy,
        preprocessor,
        postprocessor,
        expected_image_keys={"observation.images.front"},
    )

    assert adapter.is_observation_needed() is True
    observation = make_observation()
    observation["observation.images.unused"] = np.zeros((4, 5, 3), dtype=np.uint8)
    np.testing.assert_array_equal(adapter.generate_action(observation), [1.0, 2.0])
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), [1.0, 2.0])
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), [1.0, 2.0])
    assert adapter.is_observation_needed() is True

    assert preprocessor.call_count == 1
    assert postprocessor.call_count == 3
    assert len(policy.select_batches) == 3
    assert policy.select_batches[1:] == [{}, {}]
    assert policy.select_batches[0]["observation.state"].shape == (1, 3)
    assert policy.select_batches[0]["observation.state"].dtype == torch.float32
    assert policy.select_batches[0]["observation.images.front"].shape == (1, 3, 4, 5)
    assert "observation.images.unused" not in policy.select_batches[0]


def test_act_selects_runtime_camera_subset_and_ignores_unrelated_keys() -> None:
    config = SimpleNamespace(
        input_features={
            "observation.state": object(),
            "observation.images.top": object(),
            "observation.images.angle": object(),
            "observation.images.left_pillar": object(),
        },
        image_features={
            "observation.images.top": object(),
            "observation.images.angle": object(),
            "observation.images.left_pillar": object(),
        },
        env_state_feature=None,
    )

    active = _select_act_image_features(
        config,
        {
            "observation.images.top",
            "observation.images.angle",
            "observation.images.unused",
        },
    )

    assert active == {
        "observation.images.top",
        "observation.images.angle",
    }
    assert list(config.input_features) == [
        "observation.state",
        "observation.images.top",
        "observation.images.angle",
    ]


def test_act_rejects_runtime_cameras_with_no_checkpoint_match() -> None:
    config = SimpleNamespace(
        input_features={
            "observation.state": object(),
            "observation.images.top": object(),
        },
        image_features={"observation.images.top": object()},
        # The Dora runner cannot provide environment state, so an env feature
        # must not make a zero-camera configuration pass startup validation.
        env_state_feature=object(),
    )

    with pytest.raises(ValueError, match="no keys in common"):
        _select_act_image_features(
            config,
            {"observation.images.unused"},
        )


def test_standard_act_rejects_observation_while_chunk_is_queued() -> None:
    adapter = ACTPolicyAdapter(
        FakeACTPolicy(),
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
    )

    adapter.generate_action(make_observation())
    with pytest.raises(ValueError, match="still has queued actions"):
        adapter.generate_action(make_observation())


def test_standard_act_queue_stays_synchronized_when_postprocessing_fails() -> None:
    policy = FakeACTPolicy()
    postprocessor = FailOnceProcessor()
    adapter = ACTPolicyAdapter(
        policy,
        FakeProcessor(),
        postprocessor,
        expected_image_keys={"observation.images.front"},
    )

    with pytest.raises(RuntimeError, match="postprocessor failed"):
        adapter.generate_action(make_observation())

    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), [1.0, 2.0])
    assert policy.select_batches[-1] == {}


def test_temporal_ensemble_act_needs_observation_on_every_tick() -> None:
    policy = FakeACTPolicy(temporal_ensemble_coeff=0.01)
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = ACTPolicyAdapter(
        policy,
        preprocessor,
        postprocessor,
        expected_image_keys={"observation.images.front"},
    )

    np.testing.assert_array_equal(adapter.generate_action(make_observation()), [1.0, 2.0])
    assert adapter.is_observation_needed() is True
    with pytest.raises(ValueError, match="fresh observation is required"):
        adapter.generate_action({})

    np.testing.assert_array_equal(adapter.generate_action(make_observation()), [1.0, 2.0])
    assert adapter.is_observation_needed() is True
    assert preprocessor.call_count == 2
    assert postprocessor.call_count == 2


def test_act_reset_resets_policy_and_processors() -> None:
    policy = FakeACTPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = ACTPolicyAdapter(
        policy,
        preprocessor,
        postprocessor,
        expected_image_keys={"observation.images.front"},
    )

    adapter.reset()

    assert policy.reset_count == 1
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1


def test_act_policy_config_overrides_preserve_absent_and_explicit_null() -> None:
    assert _act_policy_config_overrides({}) == {}
    assert _act_policy_config_overrides({"temporal_ensemble_coeff": None}) == {
        "temporal_ensemble_coeff": None
    }


def test_act_policy_config_overrides_enable_temporal_ensemble() -> None:
    assert _act_policy_config_overrides({"temporal_ensemble_coeff": 0.01}) == {
        "temporal_ensemble_coeff": 0.01,
        "n_action_steps": 1,
    }


def test_act_policy_config_overrides_accept_negative_finite_coefficient() -> None:
    assert _act_policy_config_overrides({"temporal_ensemble_coeff": -0.01}) == {
        "temporal_ensemble_coeff": -0.01,
        "n_action_steps": 1,
    }


@pytest.mark.parametrize("value", [True, "0.01", float("inf"), float("nan")])
def test_act_policy_config_overrides_reject_invalid_coefficients(value) -> None:
    with pytest.raises(ValueError, match="finite number or null"):
        _act_policy_config_overrides({"temporal_ensemble_coeff": value})


def test_policy_loader_applies_config_overrides_before_construction(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}
    config = SimpleNamespace(type="act", device="cuda")

    class FakeLoadedPolicy:
        def to(self, device) -> None:
            captured["device"] = device

        def eval(self) -> None:
            captured["eval"] = True

    class FakePolicyClass:
        @classmethod
        def from_pretrained(cls, path, *, config):
            captured["policy_path"] = path
            captured["policy_config"] = config
            captured["policy_device"] = config.device
            return FakeLoadedPolicy()

    def fake_config_from_pretrained(path, *, cli_overrides):
        captured["config_path"] = path
        captured["cli_overrides"] = cli_overrides
        return config

    processors = (FakeProcessor(), FakeProcessor())
    monkeypatch.setattr(loader.PreTrainedConfig, "from_pretrained", fake_config_from_pretrained)
    monkeypatch.setattr(loader, "get_policy_class", lambda policy_type: FakePolicyClass)
    monkeypatch.setattr(loader, "make_pre_post_processors", lambda **kwargs: processors)

    _, preprocessor, postprocessor, loaded_config = loader.load_policy_bundle(
        tmp_path,
        device="cpu",
        policy_config_overrides={"temporal_ensemble_coeff": 0.01, "n_action_steps": 1},
        config_transform=lambda value: captured.update(config_transform=value),
    )

    assert captured["cli_overrides"] == [
        "--temporal_ensemble_coeff=0.01",
        "--n_action_steps=1",
    ]
    assert captured["policy_config"] is config
    assert captured["policy_device"] == "cpu"
    assert captured["config_transform"] is config
    assert captured["eval"] is True
    assert preprocessor is processors[0]
    assert postprocessor is processors[1]
    assert loaded_config is config


def test_runtime_image_inputs_are_filtered_by_loaded_policy_keys() -> None:
    config = PolicyNodeConfig.from_dict(
        {
            "joints": ["joint1"],
            "policy": {"type": "ACT"},
            "image_inputs": {
                "camera/top": "top",
                "camera/debug": "debug",
            },
        }
    )

    assert config.image_inputs_for({"observation.images.top"}) == {
        "camera/top": "top"
    }


def test_duplicate_camera_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="alias 必须唯一"):
        PolicyNodeConfig.from_dict(
            {
                "joints": ["joint1"],
                "policy": {"type": "ACT"},
                "image_inputs": {"image/left": "front", "image/right": "front"},
            }
        )


def test_state_and_action_joint_dimensions_can_differ() -> None:
    config = make_config(separate_state=True)

    assert config.state_joint_order == ["joint1", "joint2", "sensor_joint"]
    assert config.joint_order == ["joint1", "joint2"]
    runtime = config.runtime_policy_config()
    assert runtime["state_joint_count"] == 3
    assert runtime["action_joint_count"] == 2
    assert runtime["state_dim"] == 3
    assert runtime["action_dim"] == 2


def test_legacy_joint_config_remains_backward_compatible() -> None:
    config = make_config()

    assert config.state_joint_order == config.joint_order
    runtime = config.runtime_policy_config()
    assert runtime["state_dim"] == 2
    assert runtime["action_dim"] == 2


def test_policy_config_positional_arguments_remain_compatible() -> None:
    config = PolicyNodeConfig(
        [JointConfig(name="joint1")],
        {"type": "ACT"},
        True,
        {"image/front": "front"},
    )

    assert config.auto_start is True
    assert config.image_input_id_to_alias == {"image/front": "front"}
    assert config.state_joint_order == ["joint1"]


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        ({"joint_count": 7}, (7, 7)),
        ({"state_dim": 7}, (7, 7)),
        ({"action_dim": 7}, (7, 7)),
        ({"state_dim": 8, "action_dim": 7}, (8, 7)),
        ({"state_joint_count": 8, "action_joint_count": 7}, (8, 7)),
    ],
)
def test_act_runtime_dimension_compatibility(runtime, expected) -> None:
    assert _act_runtime_dimensions(runtime) == expected


def test_joint_command_rejects_wrong_action_length() -> None:
    config = make_config()

    with pytest.raises(ValueError, match="action length=1"):
        _build_joint_command(np.array([0.5], dtype=np.float32), config)
    with pytest.raises(ValueError, match="action length=3"):
        _build_joint_command(np.array([0.5, 0.6, 0.7], dtype=np.float32), config)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_joint_command_rejects_non_finite_actions(invalid) -> None:
    with pytest.raises(ValueError, match="finite values"):
        _build_joint_command(np.array([0.5, invalid], dtype=np.float32), make_config())


def test_joint_command_routes_each_control_mode() -> None:
    command = _build_joint_command(
        np.array([0.5, 0.6], dtype=np.float32),
        make_config(),
    )

    assert command.name == ["joint1", "joint2"]
    assert command.position == pytest.approx([0.5, 0.0])
    assert command.velocity == pytest.approx([0.0, 0.6])
    assert command.effort == pytest.approx([0.0, 0.0])
