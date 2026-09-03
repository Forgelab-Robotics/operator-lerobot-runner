from __future__ import annotations

import time
from threading import Event, Thread
from types import SimpleNamespace

import lerobot_inference.inference.policies.act.adapter as act_adapter_module
import numpy as np
import pytest
import torch
from lerobot_inference.inference.config import JointConfig, PolicyNodeConfig
from lerobot_inference.inference.main import _build_joint_command
from lerobot_inference.inference.policies import loader, registry
from lerobot_inference.inference.policies.act import (
    ACTAsyncChunkedPolicyAdapter,
    ACTPolicyAdapter,
)
from lerobot_inference.inference.policies.act.adapter import (
    _AsyncActionQueue,
    _select_act_image_features,
    _TimedAction,
)
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
        self.predict_batches: list[dict] = []
        # A non-empty private queue must not influence the adapter contract.
        self._action_queue = [torch.tensor([[99.0, 99.0]])]

    def parameters(self):
        return iter(())

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, batch: dict) -> torch.Tensor:
        self.select_batches.append(batch)
        return torch.tensor([[1.0, 2.0]])

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.predict_batches.append(batch)
        return torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            dtype=torch.float32,
        )


class BlockingFakeACTPolicy(FakeACTPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.predict_started = Event()
        self.release_prediction = Event()
        self.predict_finished = Event()

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        self.predict_batches.append(batch)
        self.predict_started.set()
        if not self.release_prediction.wait(timeout=2.0):
            raise TimeoutError("test did not release ACT prediction")
        self.predict_finished.set()
        return torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            dtype=torch.float32,
        )


def wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("condition was not met before timeout")
        time.sleep(0.01)


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


def test_async_act_rejects_temporal_ensemble() -> None:
    with pytest.raises(ValueError, match="requires inference_mode=sync"):
        ACTAsyncChunkedPolicyAdapter(
            FakeACTPolicy(temporal_ensemble_coeff=0.01),
            FakeProcessor(),
            FakeProcessor(),
            expected_image_keys={"observation.images.front"},
        )


def test_async_act_predicts_in_background_and_tick_pop_is_nonblocking() -> None:
    policy = BlockingFakeACTPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = ACTAsyncChunkedPolicyAdapter(
        policy,
        preprocessor,
        postprocessor,
        expected_image_keys={"observation.images.front"},
    )

    try:
        # The model call waits on release_prediction. A synchronous adapter would
        # block here, while async_chunked returns immediately with no ready action.
        assert adapter.generate_action(make_observation()) is None
        assert policy.predict_started.wait(timeout=1.0)
        assert adapter.is_observation_needed() is False
        assert adapter.generate_action(make_observation()) is None
        assert adapter._control_hz == 30.0
        assert adapter._actions_per_chunk == policy.config.n_action_steps
        assert adapter._chunk_size_threshold == 0.5

        policy.release_prediction.set()
        wait_until(lambda: adapter._action_queue.qsize() == 3)

        np.testing.assert_array_equal(adapter.generate_action({}), [1.0, 2.0])
        np.testing.assert_array_equal(adapter.generate_action({}), [3.0, 4.0])
        np.testing.assert_array_equal(adapter.generate_action({}), [5.0, 6.0])
        assert preprocessor.call_count == 1
        assert postprocessor.call_count == 3
        assert len(policy.predict_batches) == 1
    finally:
        policy.release_prediction.set()
        adapter.stop()


def test_async_act_single_action_chunks_advance_after_queue_depletion() -> None:
    policy = FakeACTPolicy()
    adapter = ACTAsyncChunkedPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
        actions_per_chunk=1,
    )

    try:
        first_action = adapter.generate_action(make_observation())
        if first_action is None:
            wait_until(lambda: adapter._action_queue.qsize() == 1)
            first_action = adapter.generate_action({})
        np.testing.assert_array_equal(first_action, [1.0, 2.0])
        assert adapter._action_queue.latest_action_timestep() == 0
        assert adapter._action_queue.observation_timestep() == 1
        assert adapter.is_observation_needed() is True

        second_action = adapter.generate_action(make_observation())
        if second_action is None:
            wait_until(lambda: adapter._action_queue.qsize() == 1)
            second_action = adapter.generate_action({})
        np.testing.assert_array_equal(second_action, [1.0, 2.0])
        assert adapter._action_queue.latest_action_timestep() == 1
        assert adapter._action_queue.observation_timestep() == 2
        assert len(policy.predict_batches) == 2
    finally:
        adapter.stop()


def test_async_act_queue_prefetch_threshold_and_timestep_merge() -> None:
    queue = _AsyncActionQueue()
    assert queue.observation_timestep() == 0
    first = [
        _TimedAction(0.0, 0, np.array([0.0, 0.0], dtype=np.float32)),
        _TimedAction(0.1, 1, np.array([10.0, 10.0], dtype=np.float32)),
        _TimedAction(0.2, 2, np.array([20.0, 20.0], dtype=np.float32)),
    ]
    queue.merge(first)

    assert queue.ready_for_observation(0.5) is False
    np.testing.assert_array_equal(queue.get_nowait(), [0.0, 0.0])
    assert queue.ready_for_observation(0.5) is False
    # A non-empty low-water queue keeps the latest executed timestep so
    # replacement chunks overlap queued future actions.
    assert queue.observation_timestep() == 0

    queue.merge(
        [
            _TimedAction(1.0, 0, np.array([100.0, 100.0], dtype=np.float32)),
            _TimedAction(1.1, 1, np.array([110.0, 110.0], dtype=np.float32)),
            _TimedAction(1.2, 2, np.array([120.0, 120.0], dtype=np.float32)),
        ]
    )

    # Timestep 0 was already executed and is discarded. Overlapping future
    # timesteps use LeRobot async_inference's default weighted average.
    np.testing.assert_allclose(queue.get_nowait(), [80.0, 80.0])
    assert queue.ready_for_observation(0.5) is True
    np.testing.assert_allclose(queue.get_nowait(), [90.0, 90.0])
    # Once no future action remains, the next observation advances beyond the
    # latest executed timestep instead of being rejected as a duplicate.
    assert queue.observation_timestep() == 3


def test_async_act_pause_invalidates_inflight_generation() -> None:
    policy = BlockingFakeACTPolicy()
    preprocessor = FakeProcessor()
    postprocessor = FakeProcessor()
    adapter = ACTAsyncChunkedPolicyAdapter(
        policy,
        preprocessor,
        postprocessor,
        expected_image_keys={"observation.images.front"},
    )

    try:
        assert adapter.generate_action(make_observation()) is None
        assert policy.predict_started.wait(timeout=1.0)
        generation = adapter._generation

        adapter.pause()
        assert adapter._generation > generation
        assert adapter._action_queue.qsize() == 0

        policy.release_prediction.set()
        assert policy.predict_finished.wait(timeout=1.0)
        with adapter._inference_lock:
            pass
        assert adapter._action_queue.qsize() == 0

        # Resume serializes processor/policy reset after the old inference.
        adapter.start()
        assert policy.reset_count == 1
        assert preprocessor.reset_count == 1
        assert postprocessor.reset_count == 1
    finally:
        policy.release_prediction.set()
        adapter.stop()


def test_async_act_reset_times_out_without_waiting_indefinitely(
    monkeypatch,
) -> None:
    monkeypatch.setattr(act_adapter_module, "_JOIN_TIMEOUT_S", 0.05)
    policy = BlockingFakeACTPolicy()
    adapter = ACTAsyncChunkedPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
    )

    try:
        assert adapter.generate_action(make_observation()) is None
        assert policy.predict_started.wait(timeout=1.0)

        started_at = time.monotonic()
        with pytest.raises(RuntimeError, match="reset timed out"):
            adapter.reset()
        assert time.monotonic() - started_at < 1.0
        assert adapter._needs_reset is True
        assert adapter._action_queue.qsize() == 0
        assert policy.reset_count == 0

        policy.release_prediction.set()
        assert policy.predict_finished.wait(timeout=1.0)
        with adapter._inference_lock:
            pass
        adapter.reset()
        assert adapter._needs_reset is False
        assert policy.reset_count == 1
    finally:
        policy.release_prediction.set()
        adapter.stop()


def test_async_act_resume_reports_reset_timeout_and_can_retry(monkeypatch) -> None:
    monkeypatch.setattr(act_adapter_module, "_JOIN_TIMEOUT_S", 0.05)
    policy = BlockingFakeACTPolicy()
    adapter = ACTAsyncChunkedPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
    )

    try:
        assert adapter.generate_action(make_observation()) is None
        assert policy.predict_started.wait(timeout=1.0)
        adapter.pause()

        started_at = time.monotonic()
        with pytest.raises(RuntimeError, match="reset timed out"):
            adapter.start()
        assert time.monotonic() - started_at < 1.0
        assert adapter._active is False
        assert adapter._needs_reset is True

        policy.release_prediction.set()
        assert policy.predict_finished.wait(timeout=1.0)
        with adapter._inference_lock:
            pass
        adapter.start()
        assert adapter._active is True
        assert adapter._needs_reset is False
        assert adapter._thread is not None and adapter._thread.is_alive()
        assert policy.reset_count == 1
    finally:
        policy.release_prediction.set()
        adapter.stop()


def test_async_act_stop_timeout_preserves_worker_for_safe_restart(monkeypatch) -> None:
    monkeypatch.setattr(act_adapter_module, "_JOIN_TIMEOUT_S", 0.05)
    policy = BlockingFakeACTPolicy()
    adapter = ACTAsyncChunkedPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
    )

    try:
        assert adapter.generate_action(make_observation()) is None
        assert policy.predict_started.wait(timeout=1.0)
        old_thread = adapter._thread
        assert old_thread is not None

        started_at = time.monotonic()
        adapter.stop()
        assert time.monotonic() - started_at < 1.0
        assert old_thread.is_alive()
        assert adapter._stopping is True
        assert adapter._needs_reset is True
        with pytest.raises(RuntimeError, match="still stopping"):
            adapter.start()

        policy.release_prediction.set()
        assert policy.predict_finished.wait(timeout=1.0)
        wait_until(lambda: not old_thread.is_alive())
        adapter.start()
        assert adapter._thread is not None and adapter._thread is not old_thread
        assert adapter._thread.is_alive()
        assert adapter._stopping is False
        assert adapter._needs_reset is False
        assert policy.reset_count == 1
    finally:
        policy.release_prediction.set()
        adapter.stop()


@pytest.mark.parametrize("lifecycle_method", ["reset", "stop"])
def test_async_act_reset_and_stop_invalidate_inflight_generation(
    lifecycle_method: str,
) -> None:
    policy = BlockingFakeACTPolicy()
    adapter = ACTAsyncChunkedPolicyAdapter(
        policy,
        FakeProcessor(),
        FakeProcessor(),
        expected_image_keys={"observation.images.front"},
    )
    lifecycle_finished = Event()

    try:
        assert adapter.generate_action(make_observation()) is None
        assert policy.predict_started.wait(timeout=1.0)
        generation = adapter._generation

        def run_lifecycle() -> None:
            getattr(adapter, lifecycle_method)()
            lifecycle_finished.set()

        lifecycle_thread = Thread(target=run_lifecycle)
        lifecycle_thread.start()
        wait_until(lambda: adapter._generation > generation)
        assert adapter._action_queue.qsize() == 0

        policy.release_prediction.set()
        lifecycle_thread.join(timeout=1.0)
        assert lifecycle_finished.is_set()
        assert adapter._action_queue.qsize() == 0
    finally:
        policy.release_prediction.set()
        adapter.stop()


def test_act_registry_defaults_to_sync_and_configures_async_chunked(
    monkeypatch,
) -> None:
    class StubAdapter:
        def __init__(self) -> None:
            self.validated_dimensions = None

        def validate_io_dimensions(self, state_dim: int, action_dim: int) -> None:
            self.validated_dimensions = (state_dim, action_dim)

    sync_adapter = StubAdapter()
    async_adapter = StubAdapter()
    sync_calls: list[tuple[str, dict]] = []
    async_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        registry,
        "ACTPolicyAdapter",
        SimpleNamespace(
            from_pretrained=lambda path, **kwargs: (
                sync_calls.append((path, kwargs)) or sync_adapter
            )
        ),
    )
    monkeypatch.setattr(
        registry,
        "ACTAsyncChunkedPolicyAdapter",
        SimpleNamespace(
            from_pretrained=lambda path, **kwargs: (
                async_calls.append((path, kwargs)) or async_adapter
            )
        ),
    )

    assert registry._create_act({"camera_names": []}, "/sync") is sync_adapter
    assert len(sync_calls) == 1
    assert async_calls == []

    configured = registry._create_act(
        {
            "camera_names": ["front"],
            "inference_mode": "async_chunked",
            "state_dim": 3,
            "action_dim": 2,
        },
        "/async",
    )
    assert configured is async_adapter
    assert async_calls[0][0] == "/async"
    assert async_calls[0][1]["control_hz"] == 30.0
    assert async_calls[0][1]["actions_per_chunk"] is None
    assert async_calls[0][1]["chunk_size_threshold"] == 0.5
    assert async_adapter.validated_dimensions == (3, 2)


def test_act_registry_rejects_unknown_inference_mode() -> None:
    with pytest.raises(ValueError, match="sync, async_chunked"):
        registry._create_act(
            {"camera_names": [], "inference_mode": "async"},
            "/unused",
        )


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
