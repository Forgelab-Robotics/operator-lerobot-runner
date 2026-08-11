from __future__ import annotations

import json
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from lerobot.policies.rtc import RTCConfig
from lerobot.processor import RelativeActionsProcessorStep

from lerobot_inference.inference.policies import registry
from lerobot_inference.inference.policies.pi05 import (
    PI05AsyncRTCPolicyAdapter,
    PI05PolicyAdapter,
    load_pi05_policy_strict,
)
from lerobot_inference.inference.policies.pi05 import loading as pi05_loading
from lerobot_inference.inference.policies.pi05 import sync as pi05_sync
from lerobot_inference.inference.policies.pi05.assets import (
    ensure_pi05_offline_assets,
    load_local_tokenizer,
)
from lerobot_inference.inference.policies.registry import (
    _create_pi05,
    _expected_image_keys,
    _pi05_policy_config_overrides,
)


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.reset_count = 0
        self.steps: list[object] = []

    def __call__(self, value):
        self.calls.append(value)
        return value

    def reset(self) -> None:
        self.reset_count += 1


class FailOnceProcessor(FakeProcessor):
    def __call__(self, value):
        super().__call__(value)
        if len(self.calls) == 1:
            raise RuntimeError("postprocessor failed")
        return value


class FakePI05Policy:
    def __init__(self, *, n_action_steps: int = 3) -> None:
        self.config = SimpleNamespace(
            device="cpu",
            use_amp=False,
            n_action_steps=n_action_steps,
            chunk_size=n_action_steps,
            rtc_config=None,
            input_features={
                "observation.state": SimpleNamespace(shape=(3,)),
                "observation.images.front": SimpleNamespace(shape=(3, 4, 5)),
            },
            output_features={"action": SimpleNamespace(shape=(2,))},
        )
        self.reset_count = 0
        self.select_batches: list[dict] = []
        self._native_queue: deque[torch.Tensor] = deque()

    def parameters(self):
        return iter(())

    def reset(self) -> None:
        self.reset_count += 1
        self._native_queue.clear()

    def select_action(self, batch: dict) -> torch.Tensor:
        self.select_batches.append(batch)
        if not self._native_queue:
            if not batch:
                raise RuntimeError("native queue depleted without observation")
            self._native_queue.extend(
                torch.tensor([[1.0, 2.0]])
                for _ in range(self.config.n_action_steps)
            )
        return self._native_queue.popleft()


class FakeRTCPI05Policy(FakePI05Policy):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(n_action_steps=3)
        self.config.rtc_config = RTCConfig(enabled=True, execution_horizon=2)
        self.predict_count = 0
        self.fail = fail

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        self.predict_count += 1
        if self.fail:
            raise RuntimeError("RTC model failed")
        return torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            dtype=torch.float32,
        )


class GradientRTCPI05Policy(FakeRTCPI05Policy):
    def __init__(self) -> None:
        super().__init__()
        self.guidance_count = 0

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        if kwargs.get("prev_chunk_left_over") is not None:
            with torch.enable_grad():
                latent = torch.tensor(1.0, requires_grad=True)
                denoised = latent * 2
                torch.autograd.grad(denoised, latent)
            self.guidance_count += 1
        return super().predict_action_chunk(batch, **kwargs)


class BlockingRTCPI05Policy(FakeRTCPI05Policy):
    def __init__(self) -> None:
        super().__init__()
        self.inference_started = Event()
        self.release_inference = Event()

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        self.inference_started.set()
        if not self.release_inference.wait(timeout=2.0):
            raise TimeoutError("test did not release RTC inference")
        return super().predict_action_chunk(batch, **kwargs)


class BlockingFailRTCPI05Policy(BlockingRTCPI05Policy):
    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        self.inference_started.set()
        if not self.release_inference.wait(timeout=2.0):
            raise TimeoutError("test did not release RTC inference")
        raise RuntimeError("stale generation failed")


def make_observation(*, image_dtype=np.uint8) -> dict[str, np.ndarray]:
    image = np.zeros((4, 5, 3), dtype=image_dtype)
    image.setflags(write=False)
    state = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    state.setflags(write=False)
    return {
        "observation.state": state,
        "observation.images.front": image,
    }


def make_adapter(
    *,
    policy: FakePI05Policy | None = None,
    preprocessor: FakeProcessor | None = None,
    postprocessor: FakeProcessor | None = None,
    instruction: str = "pick the cube",
) -> tuple[PI05PolicyAdapter, FakePI05Policy, FakeProcessor, FakeProcessor]:
    actual_policy = policy or FakePI05Policy()
    actual_preprocessor = preprocessor or FakeProcessor()
    actual_postprocessor = postprocessor or FakeProcessor()
    adapter = PI05PolicyAdapter(
        actual_policy,
        actual_preprocessor,
        actual_postprocessor,
        instruction=instruction,
        expected_image_keys={"observation.images.front"},
    )
    return adapter, actual_policy, actual_preprocessor, actual_postprocessor


def test_pi05_uses_native_select_action_queue() -> None:
    adapter, policy, preprocessor, postprocessor = make_adapter()

    assert adapter.is_observation_needed() is True
    np.testing.assert_array_equal(adapter.generate_action(make_observation()), [1.0, 2.0])
    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), [1.0, 2.0])
    np.testing.assert_array_equal(adapter.generate_action({}), [1.0, 2.0])
    assert adapter.is_observation_needed() is True

    assert len(preprocessor.calls) == 1
    assert len(postprocessor.calls) == 3
    assert policy.select_batches[1:] == [{}, {}]
    first_batch = policy.select_batches[0]
    assert first_batch["task"] == "pick the cube"
    assert first_batch["observation.state"].shape == (1, 3)
    assert first_batch["observation.state"].dtype == torch.float32
    assert first_batch["observation.images.front"].shape == (1, 3, 4, 5)


def test_pi05_preserves_float_images_for_native_preprocessing() -> None:
    adapter, policy, _, _ = make_adapter(policy=FakePI05Policy(n_action_steps=1))
    observation = make_observation(image_dtype=np.float32)
    image = np.full((4, 5, 3), 0.5, dtype=np.float32)
    image.setflags(write=False)
    observation["observation.images.front"] = image

    adapter.generate_action(observation)

    image = policy.select_batches[0]["observation.images.front"]
    assert image.dtype == torch.float32
    assert float(image.mean()) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("configured_dtype", "expected_dtype"),
    [("bfloat16", torch.bfloat16), ("float16", torch.float16)],
)
def test_pi05_amp_uses_model_dtype(
    monkeypatch,
    configured_dtype: str,
    expected_dtype: torch.dtype,
) -> None:
    adapter, policy, _, _ = make_adapter()
    policy.config.use_amp = True
    policy.config.dtype = configured_dtype
    adapter._device = torch.device("cuda")
    captured: dict[str, object] = {}

    def fake_autocast(*, device_type, dtype):
        captured["device_type"] = device_type
        captured["dtype"] = dtype
        return nullcontext()

    monkeypatch.setattr(pi05_sync.torch, "autocast", fake_autocast)

    with adapter._inference_autocast_context():
        pass

    assert captured == {"device_type": "cuda", "dtype": expected_dtype}


def test_pi05_amp_preserves_float32_model_precision(monkeypatch) -> None:
    adapter, policy, _, _ = make_adapter()
    policy.config.use_amp = True
    policy.config.dtype = "float32"
    adapter._device = torch.device("cuda")

    def unexpected_autocast(**kwargs):
        raise AssertionError(f"float32 model must not enable CUDA autocast: {kwargs}")

    monkeypatch.setattr(pi05_sync.torch, "autocast", unexpected_autocast)

    with adapter._inference_autocast_context():
        pass


def test_pi05_rejects_invalid_image_layout() -> None:
    adapter, _, _, _ = make_adapter()
    observation = make_observation()
    observation["observation.images.front"] = np.zeros((4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="HWC RGB"):
        adapter.generate_action(observation)


@pytest.mark.parametrize(
    "image",
    [
        np.full((4, 5, 3), 128, dtype=np.uint16),
        np.full((4, 5, 3), 128.0, dtype=np.float32),
        np.full((4, 5, 3), np.nan, dtype=np.float32),
    ],
)
def test_pi05_rejects_invalid_image_numeric_domain(image) -> None:
    adapter, _, _, _ = make_adapter()
    observation = make_observation()
    observation["observation.images.front"] = image

    with pytest.raises(ValueError, match="uint8|Floating image"):
        adapter.generate_action(observation)


def test_pi05_rejects_non_finite_state() -> None:
    adapter, _, _, _ = make_adapter()
    observation = make_observation()
    observation["observation.state"] = np.array([0.1, np.nan, 0.3])

    with pytest.raises(ValueError, match="finite values"):
        adapter.generate_action(observation)


def test_pi05_accepts_non_contiguous_inputs_by_copying() -> None:
    adapter, policy, _, _ = make_adapter(policy=FakePI05Policy(n_action_steps=1))
    observation = make_observation()
    observation["observation.state"] = np.array([0.3, 0.2, 0.1])[::-1]
    observation["observation.images.front"] = np.zeros((4, 5, 3), dtype=np.uint8)[:, ::-1]

    adapter.generate_action(observation)

    assert policy.select_batches[0]["observation.state"].is_contiguous()
    assert policy.select_batches[0]["observation.images.front"].is_contiguous()


def test_pi05_enforces_queue_observation_contract() -> None:
    adapter, _, _, _ = make_adapter()

    with pytest.raises(ValueError, match="fresh observation is required"):
        adapter.generate_action({})
    adapter.generate_action(make_observation())
    with pytest.raises(ValueError, match="still has queued actions"):
        adapter.generate_action(make_observation())


def test_pi05_queue_stays_synchronized_when_postprocessing_fails() -> None:
    adapter, policy, _, _ = make_adapter(postprocessor=FailOnceProcessor())

    with pytest.raises(RuntimeError, match="postprocessor failed"):
        adapter.generate_action(make_observation())

    assert adapter.is_observation_needed() is False
    np.testing.assert_array_equal(adapter.generate_action({}), [1.0, 2.0])
    assert policy.select_batches[-1] == {}


def test_pi05_instruction_change_discards_old_chunk() -> None:
    adapter, policy, preprocessor, postprocessor = make_adapter(instruction="task A")
    adapter.generate_action(make_observation())

    adapter.instruction = "task B"

    assert adapter.instruction == "task B"
    assert adapter.is_observation_needed() is True
    assert policy.reset_count == 1
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1
    adapter.generate_action(make_observation())
    assert policy.select_batches[-1]["task"] == "task B"


def test_pi05_pause_and_stop_reset_native_state() -> None:
    adapter, policy, preprocessor, postprocessor = make_adapter()

    adapter.pause()
    adapter.stop()

    assert policy.reset_count == 2
    assert preprocessor.reset_count == 2
    assert postprocessor.reset_count == 2
    assert adapter.is_observation_needed() is True


def test_pi05_async_rtc_is_non_blocking_and_merges_native_queue() -> None:
    policy = FakeRTCPI05Policy()
    adapter = PI05AsyncRTCPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        instruction="pick the cube",
        expected_image_keys={"observation.images.front"},
        control_hz=1.0,
        queue_threshold=0,
    )

    try:
        started_at = time.perf_counter()
        assert adapter.generate_action(make_observation()) is None
        assert time.perf_counter() - started_at < 0.5

        action = None
        deadline = time.monotonic() + 2.0
        while action is None and time.monotonic() < deadline:
            time.sleep(0.01)
            action = adapter.generate_action(make_observation())

        np.testing.assert_array_equal(action, [1.0, 2.0])
        assert policy.predict_count >= 1
        assert policy.select_batches == []
    finally:
        adapter.stop()


def test_pi05_async_rtc_allows_prefix_guidance_autograd() -> None:
    policy = GradientRTCPI05Policy()
    adapter = PI05AsyncRTCPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
        control_hz=50.0,
        queue_threshold=2,
    )

    try:
        assert adapter.generate_action(make_observation()) is None
        action = None
        deadline = time.monotonic() + 2.0
        while action is None and time.monotonic() < deadline:
            time.sleep(0.01)
            action = adapter.generate_action(make_observation())

        assert action is not None
        guidance_deadline = time.monotonic() + 2.0
        while (
            policy.predict_count < 2
            and not adapter.failed
            and time.monotonic() < guidance_deadline
        ):
            time.sleep(0.01)

        assert adapter.failed is False
        assert policy.predict_count >= 2
        assert policy.guidance_count >= 1
    finally:
        adapter.stop()


def test_pi05_async_rtc_propagates_background_errors() -> None:
    adapter = PI05AsyncRTCPolicyAdapter(
        FakeRTCPI05Policy(fail=True),
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
        control_hz=50.0,
        queue_threshold=0,
    )

    try:
        assert adapter.generate_action(make_observation()) is None
        deadline = time.monotonic() + 2.0
        while not adapter.failed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert adapter.failed is True
        with pytest.raises(SystemExit, match="async RTC inference failed"):
            adapter.generate_action(make_observation())
    finally:
        adapter.stop()


def test_pi05_async_rtc_reset_cannot_restore_stale_chunk() -> None:
    policy = BlockingRTCPI05Policy()
    adapter = PI05AsyncRTCPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
        control_hz=1.0,
        queue_threshold=0,
    )

    try:
        assert adapter.generate_action(make_observation()) is None
        assert policy.inference_started.wait(timeout=1.0)
        reset_thread = Thread(target=adapter.reset)
        reset_thread.start()
        time.sleep(0.02)
        policy.release_inference.set()
        reset_thread.join(timeout=2.0)

        assert reset_thread.is_alive() is False
        assert adapter._action_queue.qsize() == 0
    finally:
        policy.release_inference.set()
        adapter.stop()


def test_pi05_async_rtc_reset_ignores_stale_generation_failure() -> None:
    policy = BlockingFailRTCPI05Policy()
    adapter = PI05AsyncRTCPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
        control_hz=1.0,
        queue_threshold=0,
    )

    try:
        assert adapter.generate_action(make_observation()) is None
        assert policy.inference_started.wait(timeout=1.0)
        reset_thread = Thread(target=adapter.reset)
        reset_thread.start()
        time.sleep(0.02)
        policy.release_inference.set()
        reset_thread.join(timeout=2.0)

        assert reset_thread.is_alive() is False
        assert adapter.failed is False
        assert adapter._action_queue.qsize() == 0
    finally:
        policy.release_inference.set()
        adapter.stop()


def test_pi05_async_observation_snapshot_owns_its_memory() -> None:
    adapter = PI05AsyncRTCPolicyAdapter(
        FakeRTCPI05Policy(),
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
        control_hz=1.0,
        queue_threshold=0,
    )
    observation = make_observation()
    observation["observation.state"].setflags(write=True)
    observation["observation.images.front"].setflags(write=True)

    copied = adapter._copy_observation(observation)
    observation["observation.state"][0] = 99
    observation["observation.images.front"][0, 0, 0] = 255

    assert copied["observation.state"][0] != 99
    assert copied["observation.images.front"][0, 0, 0] != 255


def test_pi05_async_rtc_validates_runtime_settings() -> None:
    policy = FakeRTCPI05Policy()
    with pytest.raises(ValueError, match="control_hz must be finite and positive"):
        PI05AsyncRTCPolicyAdapter(
            policy,
            FakeProcessor(),
            FakeProcessor(),
            expected_image_keys={"observation.images.front"},
            control_hz=0,
        )
    with pytest.raises(ValueError, match="queue_threshold"):
        PI05AsyncRTCPolicyAdapter(
            policy,
            FakeProcessor(),
            FakeProcessor(),
            expected_image_keys={"observation.images.front"},
            queue_threshold=4,
        )


def test_pi05_validates_state_and_action_dimensions() -> None:
    adapter, _, _, _ = make_adapter()

    adapter.validate_io_dimensions(3, 2)
    with pytest.raises(ValueError, match="state_joints=4"):
        adapter.validate_io_dimensions(4, 2)
    with pytest.raises(ValueError, match="action_joints=3"):
        adapter.validate_io_dimensions(3, 3)


def test_pi05_relative_actions_accept_lerobot_position_feature_suffix() -> None:
    preprocessor = FakeProcessor()
    relative_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=["gripper"],
        action_names=["joint1.pos", "gripper.pos"],
    )
    preprocessor.steps = [relative_step]
    policy = FakePI05Policy()
    policy.config.action_feature_names = ["joint1.pos", "gripper.pos"]
    adapter, _, _, _ = make_adapter(policy=policy, preprocessor=preprocessor)

    adapter.configure_joint_names(
        ["joint1", "gripper"],
        ["joint1", "gripper"],
    )

    assert relative_step.action_names == ["joint1", "gripper"]
    assert relative_step._build_mask(2) == [True, False]


def test_pi05_relative_actions_validate_processor_joint_names() -> None:
    preprocessor = FakeProcessor()
    preprocessor.steps = [
        RelativeActionsProcessorStep(
            enabled=True,
            exclude_joints=["gripper"],
            action_names=["trained_joint.pos", "gripper.pos"],
        )
    ]
    adapter, _, _, _ = make_adapter(preprocessor=preprocessor)

    with pytest.raises(ValueError, match="processor action_names"):
        adapter.configure_joint_names(
            ["runtime_joint", "gripper"],
            ["runtime_joint", "gripper"],
        )


def test_pi05_from_pretrained_injects_local_tokenizer_and_strict_loader(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    tokenizer = object()
    policy = FakePI05Policy()
    config = SimpleNamespace(
        type="pi05",
        empty_cameras=1,
        image_features={
            "observation.images.front": object(),
            "observation.images.empty_camera_0": object(),
        },
    )
    processors = (FakeProcessor(), FakeProcessor())

    monkeypatch.setattr(
        pi05_sync,
        "ensure_pi05_offline_assets",
        lambda pretrained_path, tokenizer_path: (Path("/policy"), "/tokenizer"),
    )
    monkeypatch.setattr(pi05_sync, "load_local_tokenizer", lambda path: tokenizer)

    def fake_load_policy_bundle(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return policy, processors[0], processors[1], config

    monkeypatch.setattr(pi05_sync, "load_policy_bundle", fake_load_policy_bundle)

    adapter = PI05PolicyAdapter.from_pretrained(
        "/policy",
        tokenizer_path="/tokenizer",
        expected_image_keys={"observation.images.front"},
    )

    assert isinstance(adapter, PI05PolicyAdapter)
    assert captured["preprocessor_overrides"] == {
        "tokenizer_processor": {
            "tokenizer_name": None,
            "tokenizer": tokenizer,
        }
    }
    assert captured["policy_loader"] is load_pi05_policy_strict


def test_pi05_from_pretrained_allows_subset_and_ignores_extra_camera(monkeypatch) -> None:
    policy = FakePI05Policy()
    config = SimpleNamespace(
        type="pi05",
        image_features={
            "observation.images.front": object(),
            "observation.images.wrist": object(),
        },
    )
    monkeypatch.setattr(
        pi05_sync,
        "ensure_pi05_offline_assets",
        lambda pretrained_path, tokenizer_path: (Path("/policy"), "/tokenizer"),
    )
    monkeypatch.setattr(pi05_sync, "load_local_tokenizer", lambda path: object())
    monkeypatch.setattr(
        pi05_sync,
        "load_policy_bundle",
        lambda path, **kwargs: (policy, FakeProcessor(), FakeProcessor(), config),
    )

    adapter = PI05PolicyAdapter.from_pretrained(
        "/policy",
        tokenizer_path="/tokenizer",
        expected_image_keys={
            "observation.images.wrist",
            "observation.images.debug",
        },
    )

    assert adapter.required_image_keys == frozenset({"observation.images.wrist"})
    assert policy._preprocess_images.__func__ is pi05_sync._preprocess_images_in_checkpoint_order


def test_pi05_missing_camera_keeps_checkpoint_slot_order() -> None:
    front_key = "observation.images.front"
    wrist_key = "observation.images.wrist"

    class FakeImagePolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                image_features={front_key: object(), wrist_key: object()},
                image_resolution=(4, 5),
            )
            self.parameter = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            return iter((self.parameter,))

    wrist = torch.full((1, 3, 4, 5), 0.75)
    images, masks = pi05_sync._preprocess_images_in_checkpoint_order(
        FakeImagePolicy(),
        {wrist_key: wrist},
    )

    assert len(images) == 2
    assert torch.all(images[0] == -1)
    assert masks[0].tolist() == [False]
    torch.testing.assert_close(images[1], torch.full_like(wrist, 0.5))
    assert masks[1].tolist() == [True]


def test_pi05_from_pretrained_rejects_camera_mismatch(monkeypatch) -> None:
    policy = FakePI05Policy()
    config = SimpleNamespace(
        type="pi05",
        image_features={"observation.images.front": object()},
    )
    monkeypatch.setattr(
        pi05_sync,
        "ensure_pi05_offline_assets",
        lambda pretrained_path, tokenizer_path: (Path("/policy"), "/tokenizer"),
    )
    monkeypatch.setattr(pi05_sync, "load_local_tokenizer", lambda path: object())
    monkeypatch.setattr(
        pi05_sync,
        "load_policy_bundle",
        lambda path, **kwargs: (policy, FakeProcessor(), FakeProcessor(), config),
    )

    with pytest.raises(ValueError, match="no keys in common"):
        PI05PolicyAdapter.from_pretrained(
            "/policy",
            tokenizer_path="/tokenizer",
            expected_image_keys={"observation.images.wrist"},
        )


def test_explicit_image_keys_can_select_dora_alias_subset() -> None:
    assert _expected_image_keys(
        {"expected_image_keys": ["observation.images.front"]},
        ["front", "debug"],
    ) == {"observation.images.front"}


def test_explicit_image_keys_cannot_bypass_dora_aliases() -> None:
    with pytest.raises(ValueError, match="must be produced by image_inputs"):
        _expected_image_keys(
            {"expected_image_keys": ["observation.images.front"]},
            ["wrist"],
        )


def test_pi05_async_rtc_config_overrides_are_nested() -> None:
    assert _pi05_policy_config_overrides(
        {
            "compile_model": False,
            "rtc": {
                "execution_horizon": 10,
                "max_guidance_weight": 8.0,
                "prefix_attention_schedule": "LINEAR",
            },
        },
        async_rtc=True,
    ) == {
        "compile_model": False,
        "rtc_config.enabled": True,
        "rtc_config.prefix_attention_schedule": "LINEAR",
        "rtc_config.max_guidance_weight": 8.0,
        "rtc_config.execution_horizon": 10,
    }


def test_pi05_async_rtc_rejects_unknown_config_and_compile() -> None:
    with pytest.raises(ValueError, match="Unknown policy.rtc keys"):
        _pi05_policy_config_overrides(
            {"rtc": {"execution_horizn": 10}},
            async_rtc=True,
        )
    with pytest.raises(ValueError, match="compile_model=true"):
        _pi05_policy_config_overrides(
            {"compile_model": True},
            async_rtc=True,
        )


def test_pi05_registry_selects_async_rtc_adapter(monkeypatch) -> None:
    adapter = SimpleNamespace(
        validate_io_dimensions=lambda *args: None,
        configure_joint_names=lambda *args: None,
    )
    captured: dict[str, object] = {}

    def fake_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(
        registry.PI05AsyncRTCPolicyAdapter,
        "from_pretrained",
        fake_from_pretrained,
    )

    assert (
        _create_pi05(
            {
                "tokenizer_path": "/unused",
                "inference_mode": "async_rtc",
                "control_hz": 50,
                "rtc": {"execution_horizon": 10, "queue_threshold": 20},
            },
            "/unused",
        )
        is adapter
    )
    assert captured["control_hz"] == 50.0
    assert captured["queue_threshold"] == 20
    assert cast(dict, captured["policy_config_overrides"])["rtc_config.enabled"] is True


def test_removed_pi05_async_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonzero policy.get_actions_threshold"):
        _create_pi05(
            {
                "tokenizer_path": "/unused",
                "get_actions_threshold": 1,
            },
            "/unused",
        )


def test_legacy_zero_pi05_async_threshold_is_ignored(monkeypatch) -> None:
    adapter = SimpleNamespace(
        validate_io_dimensions=lambda *args: None,
        configure_joint_names=lambda *args: None,
    )
    monkeypatch.setattr(
        registry.PI05PolicyAdapter,
        "from_pretrained",
        lambda *args, **kwargs: adapter,
    )

    assert (
        _create_pi05(
            {
                "tokenizer_path": "/unused",
                "get_actions_threshold": 0,
            },
            "/unused",
        )
        is adapter
    )


def write_pi05_assets(policy_dir: Path, tokenizer_dir: Path, *, obsolete: bool = False) -> None:
    policy_dir.mkdir()
    tokenizer_dir.mkdir()
    (policy_dir / "config.json").write_text("{}", encoding="utf-8")
    (policy_dir / "model.safetensors").write_bytes(b"weights")
    steps = [{"registry_name": "delta_actions_processor"}] if obsolete else []
    (policy_dir / "policy_preprocessor.json").write_text(
        json.dumps({"steps": steps}), encoding="utf-8"
    )
    (policy_dir / "policy_postprocessor.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_pi05_offline_asset_validation_does_not_modify_checkpoint(tmp_path) -> None:
    policy_dir = tmp_path / "policy"
    tokenizer_dir = tmp_path / "tokenizer"
    write_pi05_assets(policy_dir, tokenizer_dir)
    before = {
        path.name: path.read_bytes()
        for path in policy_dir.iterdir()
        if path.is_file()
    }

    resolved_policy, resolved_tokenizer = ensure_pi05_offline_assets(
        policy_dir, str(tokenizer_dir)
    )

    after = {
        path.name: path.read_bytes()
        for path in policy_dir.iterdir()
        if path.is_file()
    }
    assert resolved_policy == policy_dir.resolve()
    assert resolved_tokenizer == str(tokenizer_dir.resolve())
    assert after == before


def test_pi05_obsolete_processor_requires_explicit_migration(tmp_path) -> None:
    policy_dir = tmp_path / "policy"
    tokenizer_dir = tmp_path / "tokenizer"
    write_pi05_assets(policy_dir, tokenizer_dir, obsolete=True)

    with pytest.raises(ValueError, match="must be migrated"):
        ensure_pi05_offline_assets(policy_dir, str(tokenizer_dir))
    assert "delta_actions_processor" in (
        policy_dir / "policy_preprocessor.json"
    ).read_text(encoding="utf-8")


def test_local_tokenizer_loading_is_offline_and_instance_scoped(monkeypatch) -> None:
    captured: dict[str, object] = {}
    tokenizer = object()

    def fake_from_pretrained(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return tokenizer

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    assert load_local_tokenizer("/local/tokenizer") is tokenizer
    assert captured == {"path": "/local/tokenizer", "local_files_only": True}


def make_strict_config(*, rtc_config=None, n_action_steps=3, chunk_size=3):
    return SimpleNamespace(
        rtc_config=rtc_config,
        n_action_steps=n_action_steps,
        chunk_size=chunk_size,
    )


def test_pi05_strict_loader_rejects_corrupt_weights(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not safetensors")

    class FakePolicyClass:
        def __new__(cls, config):
            return SimpleNamespace()

    with pytest.raises(RuntimeError, match="Failed to load PI0.5 weights"):
        load_pi05_policy_strict(
            FakePolicyClass,
            tmp_path,
            make_strict_config(),
        )


def test_pi05_strict_loader_rejects_incompatible_weights(monkeypatch, tmp_path) -> None:
    class FakePolicy:
        def _fix_pytorch_state_dict_keys(self, state_dict, config):
            return state_dict

        def load_state_dict(self, state_dict, *, strict):
            raise RuntimeError("shape mismatch")

    class FakePolicyClass:
        def __new__(cls, config):
            return FakePolicy()

    monkeypatch.setattr(
        pi05_loading,
        "load_file",
        lambda path: {"weight": torch.zeros(1)},
    )

    with pytest.raises(RuntimeError, match="incompatible with config"):
        load_pi05_policy_strict(
            FakePolicyClass,
            tmp_path,
            make_strict_config(),
        )


def test_pi05_strict_loader_rejects_ambiguous_remapped_keys(monkeypatch, tmp_path) -> None:
    class FakePolicy:
        def _fix_pytorch_state_dict_keys(self, state_dict, config):
            return state_dict

        def load_state_dict(self, state_dict, *, strict):
            raise AssertionError("load_state_dict must not be called after a collision")

    class FakePolicyClass:
        def __new__(cls, config):
            return FakePolicy()

    monkeypatch.setattr(
        pi05_loading,
        "load_file",
        lambda path: {
            "weight": torch.zeros(1),
            "model.weight": torch.ones(1),
        },
    )

    with pytest.raises(RuntimeError, match="incompatible with config") as exc_info:
        load_pi05_policy_strict(
            FakePolicyClass,
            tmp_path,
            make_strict_config(),
        )
    assert "Ambiguous PI0.5 weight keys" in str(exc_info.value.__cause__)


def test_pi05_strict_loader_remaps_standard_keys(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakePolicy:
        def _fix_pytorch_state_dict_keys(self, state_dict, config):
            return state_dict

        def load_state_dict(self, state_dict, *, strict):
            captured["state_dict"] = state_dict
            captured["strict"] = strict

    class FakePolicyClass:
        def __new__(cls, config):
            return FakePolicy()

    monkeypatch.setattr(
        pi05_loading,
        "load_file",
        lambda path: {
            "weight": torch.zeros(1),
            "model.bias": torch.ones(1),
        },
    )

    policy = load_pi05_policy_strict(
        FakePolicyClass,
        tmp_path,
        make_strict_config(),
    )

    assert isinstance(policy, FakePolicy)
    assert set(cast(dict, captured["state_dict"])) == {"model.weight", "model.bias"}
    assert captured["strict"] is True


def test_pi05_strict_loader_allows_enabled_rtc_backend(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not safetensors")

    class FakePolicyClass:
        def __new__(cls, config):
            return SimpleNamespace()

    with pytest.raises(RuntimeError, match="Failed to load PI0.5 weights"):
        load_pi05_policy_strict(
            FakePolicyClass,
            tmp_path,
            make_strict_config(rtc_config=SimpleNamespace(enabled=True)),
        )


def test_pi05_strict_loader_allows_disabled_rtc_backend(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"not safetensors")

    class FakePolicyClass:
        def __new__(cls, config):
            return SimpleNamespace()

    with pytest.raises(RuntimeError, match="Failed to load PI0.5 weights"):
        load_pi05_policy_strict(
            FakePolicyClass,
            tmp_path,
            make_strict_config(rtc_config=SimpleNamespace(enabled=False)),
        )


@pytest.mark.parametrize("n_action_steps", [0, -1, 1.5, True])
def test_pi05_strict_loader_rejects_invalid_action_steps(tmp_path, n_action_steps) -> None:
    with pytest.raises(ValueError, match="1 <= n_action_steps <= chunk_size"):
        load_pi05_policy_strict(
            object,
            tmp_path,
            make_strict_config(n_action_steps=n_action_steps),
        )
