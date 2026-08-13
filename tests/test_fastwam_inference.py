from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot_inference.inference.config import PolicyNodeConfig
from lerobot_inference.inference.policies import registry
from lerobot_inference.inference.policies.fastwam import (
    FastWAMPolicyAdapter,
    ensure_fastwam_offline_assets,
    load_fastwam_policy_strict,
)
from lerobot_inference.inference.policies.registry import (
    _create_fastwam,
    _fastwam_policy_config_overrides,
    normalize_policy_type,
)


class FakeProcessor:
    def __init__(self, name: str, order: list[str] | None = None) -> None:
        self.name = name
        self.order = order if order is not None else []
        self.calls: list[object] = []
        self.reset_count = 0

    def __call__(self, value):
        self.order.append(self.name)
        self.calls.append(value)
        return value

    def reset(self) -> None:
        self.reset_count += 1


class FailOnceProcessor(FakeProcessor):
    def __call__(self, value):
        result = super().__call__(value)
        if len(self.calls) == 1:
            raise RuntimeError("postprocessor failed")
        return result


class FakeFastWAMPolicy:
    def __init__(self, *, action: torch.Tensor | None = None) -> None:
        self.config = SimpleNamespace(
            type="fastwam",
            device="cpu",
            use_amp=False,
            torch_dtype="bfloat16",
            n_action_steps=3,
            input_features={
                "observation.state": SimpleNamespace(shape=(8,)),
                "observation.images.image": SimpleNamespace(shape=(3, 224, 224)),
                "observation.images.image2": SimpleNamespace(shape=(3, 224, 224)),
            },
            output_features={"action": SimpleNamespace(shape=(7,))},
        )
        self.reset_count = 0
        self.select_batches: list[dict] = []
        self._queue: deque[torch.Tensor] = deque()
        self._action = action if action is not None else torch.arange(7, dtype=torch.float32)[None]

    def parameters(self):
        return iter(())

    def reset(self) -> None:
        self.reset_count += 1
        self._queue.clear()

    def select_action(self, batch: dict) -> torch.Tensor:
        self.select_batches.append(batch)
        if not self._queue:
            if not batch:
                raise RuntimeError("queue depleted")
            self._queue.extend(self._action.clone() for _ in range(3))
        return self._queue.popleft()


def make_observation() -> dict[str, np.ndarray]:
    state = np.arange(8, dtype=np.float64)[::-1]
    image = np.zeros((224, 224, 3), dtype=np.uint8)[:, ::-1]
    image2 = np.ones((224, 224, 3), dtype=np.float32) * 0.5
    state.setflags(write=False)
    image.setflags(write=False)
    return {
        "observation.state": state,
        "observation.images.image": image,
        "observation.images.image2": image2,
    }


def make_adapter(
    *,
    policy: FakeFastWAMPolicy | None = None,
    postprocessor: FakeProcessor | None = None,
) -> tuple[FastWAMPolicyAdapter, FakeFastWAMPolicy, FakeProcessor, FakeProcessor, list[str]]:
    order: list[str] = []
    actual_policy = policy or FakeFastWAMPolicy()
    preprocessor = FakeProcessor("pre", order)
    actual_postprocessor = postprocessor or FakeProcessor("post", order)
    actual_postprocessor.order = order
    adapter = FastWAMPolicyAdapter(
        actual_policy,
        preprocessor,
        actual_postprocessor,
        instruction="pick up the object",
        expected_image_keys={
            "observation.images.image",
            "observation.images.image2",
        },
    )
    return adapter, actual_policy, preprocessor, actual_postprocessor, order


@pytest.mark.parametrize("alias", ["fastwam", "FastWAM", "fast-wam"])
def test_fastwam_registry_aliases(alias: str) -> None:
    assert normalize_policy_type(alias) == "fastwam"


def test_fastwam_uses_processor_and_native_queue() -> None:
    adapter, policy, preprocessor, postprocessor, order = make_adapter()

    first = adapter.generate_action(make_observation())
    second = adapter.generate_action({})
    third = adapter.generate_action({})

    np.testing.assert_array_equal(first, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(third, first)
    assert adapter.is_observation_needed() is True
    assert order == ["pre", "post", "post", "post"]
    assert len(preprocessor.calls) == 1
    assert len(postprocessor.calls) == 3
    assert policy.select_batches[1:] == [{}, {}]
    batch = policy.select_batches[0]
    assert batch["task"] == "pick up the object"
    assert batch["observation.state"].dtype == torch.float32
    assert batch["observation.state"].is_contiguous()
    assert batch["observation.images.image"].is_contiguous()


def test_fastwam_requires_exact_image_keys() -> None:
    adapter, *_ = make_adapter()
    observation = make_observation()
    observation.pop("observation.images.image2")
    with pytest.raises(ValueError, match="exactly match"):
        adapter.generate_action(observation)

    observation = make_observation()
    observation["observation.images.debug"] = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly match"):
        adapter.generate_action(observation)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((224, 224), dtype=np.uint8),
        np.zeros((224, 224, 3), dtype=np.uint16),
        np.full((224, 224, 3), 2.0, dtype=np.float32),
        np.full((224, 224, 3), np.nan, dtype=np.float32),
    ],
)
def test_fastwam_rejects_invalid_images(value: np.ndarray) -> None:
    adapter, *_ = make_adapter()
    observation = make_observation()
    observation["observation.images.image"] = value
    with pytest.raises(ValueError, match="HWC RGB|uint8|Floating image"):
        adapter.generate_action(observation)


def test_fastwam_validates_state_and_output() -> None:
    adapter, *_ = make_adapter()
    observation = make_observation()
    observation["observation.state"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError, match="state shape"):
        adapter.generate_action(observation)

    adapter, *_ = make_adapter(
        policy=FakeFastWAMPolicy(action=torch.tensor([[1, 2, 3, 4, 5, 6, float("nan")]]))
    )
    with pytest.raises(ValueError, match="finite"):
        adapter.generate_action(make_observation())


def test_fastwam_queue_stays_synchronized_after_postprocess_failure() -> None:
    failing = FailOnceProcessor("post")
    adapter, policy, *_ = make_adapter(postprocessor=failing)
    with pytest.raises(RuntimeError, match="postprocessor failed"):
        adapter.generate_action(make_observation())

    assert adapter.is_observation_needed() is False
    adapter.generate_action({})
    assert policy.select_batches[-1] == {}


def test_fastwam_lifecycle_resets_every_component() -> None:
    adapter, policy, preprocessor, postprocessor, _ = make_adapter()
    adapter.reset()
    adapter.pause()
    adapter.stop()
    assert policy.reset_count == 3
    assert preprocessor.reset_count == 3
    assert postprocessor.reset_count == 3


def test_fastwam_dimension_validation() -> None:
    adapter, *_ = make_adapter()
    adapter.validate_io_dimensions(8, 7)
    with pytest.raises(ValueError, match="runtime dimensions"):
        adapter.validate_io_dimensions(7, 7)


def test_fastwam_robotwin_single_composite_image_and_14d_action() -> None:
    policy = FakeFastWAMPolicy(action=torch.arange(14, dtype=torch.float32)[None])
    policy.config.input_features = {
        "observation.state": SimpleNamespace(shape=(14,)),
        "observation.images.image": SimpleNamespace(shape=(3, 384, 320)),
    }
    policy.config.output_features = {"action": SimpleNamespace(shape=(14,))}
    policy.config.n_action_steps = 1
    adapter = FastWAMPolicyAdapter(
        policy,
        FakeProcessor("pre"),
        FakeProcessor("post"),
        instruction="complete the manipulation task",
        expected_image_keys={"observation.images.image"},
    )
    observation = {
        "observation.state": np.zeros(14, dtype=np.float64),
        "observation.images.image": np.zeros((384, 320, 3), dtype=np.uint8),
    }

    adapter.validate_io_dimensions(14, 14)
    action = adapter.generate_action(observation)

    np.testing.assert_array_equal(action, np.arange(14, dtype=np.float32))
    batch = policy.select_batches[0]
    assert batch["observation.state"].shape == (1, 14)
    assert batch["observation.images.image"].shape == (1, 3, 384, 320)


def test_fastwam_config_overrides_are_validated() -> None:
    assert _fastwam_policy_config_overrides(
        {
            "n_action_steps": 10,
            "num_inference_steps": 5,
            "torch_dtype": "bfloat16",
            "use_amp": False,
            "sigma_shift": None,
        }
    ) == {
        "n_action_steps": 10,
        "num_inference_steps": 5,
        "sigma_shift": None,
        "use_amp": False,
        "torch_dtype": "bfloat16",
    }
    with pytest.raises(ValueError, match="positive integer"):
        _fastwam_policy_config_overrides({"n_action_steps": 0})
    with pytest.raises(ValueError, match="positive integer"):
        _fastwam_policy_config_overrides({"n_action_steps": 1.5})
    with pytest.raises(ValueError, match="torch_dtype"):
        _fastwam_policy_config_overrides({"torch_dtype": "int8"})


def _write_fastwam_assets(root: Path) -> tuple[Path, Path, Path]:
    checkpoint = root / "checkpoint"
    wan = root / "wan"
    tokenizer = root / "tokenizer"
    checkpoint.mkdir()
    (wan / "vae").mkdir(parents=True)
    (wan / "text_encoder").mkdir()
    tokenizer.mkdir()

    for name in ("config.json", "model.safetensors"):
        (checkpoint / name).write_text("{}")
    processor = {
        "steps": [{"registry_name": "normalizer_processor", "state_file": "stats.safetensors"}]
    }
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (checkpoint / name).write_text(json.dumps(processor))
    (checkpoint / "stats.safetensors").write_text("")
    for path in (
        wan / "vae" / "config.json",
        wan / "vae" / "diffusion_pytorch_model.safetensors",
        wan / "text_encoder" / "config.json",
        wan / "text_encoder" / "model.safetensors.index.json",
        wan / "text_encoder" / "model-00001-of-00001.safetensors",
        tokenizer / "tokenizer_config.json",
        tokenizer / "special_tokens_map.json",
        tokenizer / "spiece.model",
    ):
        path.write_text("{}")
    return checkpoint, wan, tokenizer


def test_fastwam_offline_asset_validation(tmp_path: Path) -> None:
    checkpoint, wan, tokenizer = _write_fastwam_assets(tmp_path)
    assert ensure_fastwam_offline_assets(checkpoint, wan, tokenizer) == (
        checkpoint.resolve(),
        wan.resolve(),
        tokenizer.resolve(),
    )
    (wan / "vae" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="VAE weights"):
        ensure_fastwam_offline_assets(checkpoint, wan, tokenizer)


@pytest.mark.parametrize(
    ("root_name", "relative_path", "message"),
    [
        ("checkpoint", "model.safetensors", "model.safetensors"),
        ("checkpoint", "policy_postprocessor.json", "policy_postprocessor"),
        ("wan", "text_encoder/config.json", "encoder config"),
        ("tokenizer", "tokenizer_config.json", "tokenizer config"),
    ],
)
def test_fastwam_reports_missing_offline_assets(
    tmp_path: Path,
    root_name: str,
    relative_path: str,
    message: str,
) -> None:
    checkpoint, wan, tokenizer = _write_fastwam_assets(tmp_path)
    roots = {"checkpoint": checkpoint, "wan": wan, "tokenizer": tokenizer}
    (roots[root_name] / relative_path).unlink()
    with pytest.raises(FileNotFoundError, match=message):
        ensure_fastwam_offline_assets(checkpoint, wan, tokenizer)


def test_fastwam_asset_validation_never_calls_hub(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    checkpoint, wan, tokenizer = _write_fastwam_assets(tmp_path)

    def unexpected_download(*args, **kwargs):
        raise AssertionError(f"FastWAM offline validation accessed Hub: {args}, {kwargs}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_download)
    ensure_fastwam_offline_assets(checkpoint, wan, tokenizer)


def test_fastwam_strict_loader_uses_local_vae_source(tmp_path: Path) -> None:
    from lerobot.policies.fastwam.wan import components

    captured: dict[str, object] = {}
    original = components.WAN22_DIFFUSERS_MODEL_ID

    class FakePolicyClass:
        @classmethod
        def from_pretrained(cls, path, *, config, strict):
            captured.update(
                path=path,
                config=config,
                strict=strict,
                source=components.WAN22_DIFFUSERS_MODEL_ID,
            )
            return "policy"

    config = SimpleNamespace()
    result = load_fastwam_policy_strict(
        FakePolicyClass,
        tmp_path,
        config,
        wan_diffusers_path=tmp_path / "wan",
    )
    assert result == "policy"
    assert captured["strict"] is True
    assert captured["source"] == str((tmp_path / "wan").resolve())
    assert components.WAN22_DIFFUSERS_MODEL_ID == original


def test_fastwam_registry_factory(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_adapter = SimpleNamespace(
        validate_io_dimensions=lambda state_dim, action_dim: captured.update(
            dimensions=(state_dim, action_dim)
        )
    )

    def fake_from_pretrained(path, **kwargs):
        captured.update(path=path, kwargs=kwargs)
        return fake_adapter

    monkeypatch.setattr(registry.FastWAMPolicyAdapter, "from_pretrained", fake_from_pretrained)
    result = _create_fastwam(
        {
            "wan_diffusers_path": "/wan",
            "tokenizer_path": "/tokenizer",
            "instruction": "pick",
            "camera_names": ["image", "image2"],
            "state_joint_count": 8,
            "action_joint_count": 7,
        },
        "/checkpoint",
    )
    assert result is fake_adapter
    assert captured["dimensions"] == (8, 7)
    assert captured["kwargs"]["expected_image_keys"] == {
        "observation.images.image",
        "observation.images.image2",
    }


def test_fastwam_robotwin_config_matches_checkpoint_contract() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config/inference/fastwam_robotwin.yaml"
    config = PolicyNodeConfig.from_yaml_path(config_path)
    runtime = config.runtime_policy_config()

    assert len(config.state_joint_order) == 14
    assert len(config.joint_order) == 14
    assert config.image_input_id_to_alias == {"image/combined": "image"}
    assert Path(runtime["pretrained_path"]).name == "fastwam_robotwin_uncond_3cam_384"
    assert runtime["instruction"] == "complete the manipulation task"
