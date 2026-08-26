"""LeRobot ACT policy adapter for Dora inference."""

from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Condition, Lock, Thread
from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from lerobot_inference.inference.compile_utils import maybe_compile_policy
from lerobot_inference.inference.observation import action_tensor_to_numpy
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.loader import load_policy_bundle

logger = logging.getLogger(__name__)

_IMAGE_KEY_PREFIX = "observation.images."
_JOIN_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class _TimedObservation:
    timestamp: float
    timestep: int
    generation: int
    observation: dict[str, np.ndarray]


@dataclass(frozen=True)
class _TimedAction:
    timestamp: float
    timestep: int
    action: np.ndarray


class _AsyncActionQueue:
    """Thread-safe timestep queue matching LeRobot async client merge semantics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._actions: dict[int, _TimedAction] = {}
        self._latest_action = -1
        self._action_chunk_size = 0

    def clear(self) -> None:
        with self._lock:
            self._actions.clear()
            self._latest_action = -1
            self._action_chunk_size = 0

    def qsize(self) -> int:
        with self._lock:
            return len(self._actions)

    def ready_for_observation(self, threshold: float) -> bool:
        with self._lock:
            if self._action_chunk_size <= 0:
                return True
            return len(self._actions) / self._action_chunk_size <= threshold

    def latest_action_timestep(self) -> int:
        with self._lock:
            return self._latest_action

    def observation_timestep(self) -> int:
        with self._lock:
            if self._actions:
                # Keep overlap while queued actions remain so a replacement chunk
                # can be merged into future timesteps that have not executed yet.
                return max(self._latest_action, 0)
            # Once the queue is empty, the next prediction must start after the
            # latest executed action. This also keeps one-action chunks progressing.
            return max(self._latest_action + 1, 0)

    def get_nowait(self) -> np.ndarray | None:
        with self._lock:
            if not self._actions:
                return None
            timestep = min(self._actions)
            timed_action = self._actions.pop(timestep)
            self._latest_action = timestep
            return timed_action.action

    def merge(self, incoming_actions: list[_TimedAction]) -> None:
        if not incoming_actions:
            return
        with self._lock:
            self._action_chunk_size = max(
                self._action_chunk_size,
                len(incoming_actions),
            )
            for incoming in incoming_actions:
                if incoming.timestep <= self._latest_action:
                    continue
                current = self._actions.get(incoming.timestep)
                if current is None:
                    self._actions[incoming.timestep] = incoming
                    continue
                # LeRobot 0.6.1 async_inference defaults to weighted_average.
                merged = 0.3 * current.action + 0.7 * incoming.action
                self._actions[incoming.timestep] = _TimedAction(
                    timestamp=incoming.timestamp,
                    timestep=incoming.timestep,
                    action=merged.astype(np.float32, copy=False),
                )


def _select_act_image_features(
    config: Any,
    runtime_image_keys: set[str] | None,
) -> set[str]:
    """Restrict ACT visual features to runtime cameras present in the checkpoint."""
    model_image_keys = set(config.image_features)
    if runtime_image_keys is None:
        return model_image_keys

    active_image_keys = model_image_keys & runtime_image_keys
    ignored_runtime_keys = runtime_image_keys - model_image_keys
    omitted_model_keys = model_image_keys - runtime_image_keys

    if ignored_runtime_keys:
        logger.warning(
            "Ignoring image inputs not used by the ACT checkpoint: %s",
            sorted(ignored_runtime_keys),
        )
    if omitted_model_keys:
        logger.warning(
            "Omitting ACT checkpoint image features not provided at runtime: %s",
            sorted(omitted_model_keys),
        )

    if not active_image_keys and model_image_keys:
        raise ValueError(
            "ACT runtime image inputs have no keys in common with the checkpoint: "
            f"runtime={sorted(runtime_image_keys)}, checkpoint={sorted(model_image_keys)}"
        )

    config.input_features = {
        key: feature
        for key, feature in config.input_features.items()
        if key not in omitted_model_keys
    }
    return active_image_keys


class ACTPolicyAdapter(LerobotPolicyAdapter):
    """Chunked ACT inference via LeRobot select_action + processor pipelines."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        expected_image_keys: set[str],
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._expected_image_keys = expected_image_keys
        self._temporal_ensemble = policy.config.temporal_ensemble_coeff is not None
        self._queued_steps_remaining = 0
        try:
            self._device = next(policy.parameters()).device
        except StopIteration:
            self._device = torch.device(getattr(policy.config, "device", "cpu") or "cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        device: str | None = None,
        expected_image_keys: set[str] | None = None,
        torch_compile: bool = False,
        policy_config_overrides: dict[str, Any] | None = None,
    ) -> ACTPolicyAdapter:
        runtime_image_keys = (
            set(expected_image_keys) if expected_image_keys is not None else None
        )
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
            policy_config_overrides=policy_config_overrides,
            config_transform=lambda config: _select_act_image_features(
                config,
                runtime_image_keys,
            ),
        )
        if config.type != "act":
            raise ValueError(f"ACTPolicyAdapter expects type act, got {config.type!r}")
        policy = maybe_compile_policy(policy, enabled=torch_compile)
        model_image_keys = set(config.image_features)
        return cls(
            policy,
            preprocessor,
            postprocessor,
            expected_image_keys=model_image_keys,
        )

    @property
    def required_image_keys(self) -> frozenset[str]:
        return frozenset(self._expected_image_keys)

    def validate_io_dimensions(self, state_dim: int, action_dim: int) -> None:
        """Validate observation and command dimensions independently."""
        state_feat = self._policy.config.input_features.get("observation.state")
        action_feat = self._policy.config.output_features.get("action")
        if state_feat is None or action_feat is None:
            raise ValueError("ACT config missing observation.state or action features")
        model_state_dim = int(state_feat.shape[0])
        model_action_dim = int(action_feat.shape[0])
        if model_state_dim != state_dim or model_action_dim != action_dim:
            raise ValueError(
                "ACT runtime dimensions do not match checkpoint: "
                f"state_joints={state_dim}, model_state_dim={model_state_dim}, "
                f"action_joints={action_dim}, model_action_dim={model_action_dim}"
            )

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._queued_steps_remaining = 0

    def is_observation_needed(self) -> bool:
        # Temporal ensembling predicts from every observation.  Standard ACT only
        # needs a new observation when LeRobot's select_action queue is depleted.
        return self._temporal_ensemble or self._queued_steps_remaining == 0

    def _validate_observation(self, observation: dict[str, Any]) -> None:
        missing = [key for key in sorted(self._expected_image_keys) if key not in observation]
        if missing:
            raise KeyError(f"Missing camera observations: {missing}")
        if "observation.state" not in observation and any(
            key == "observation.state" for key in self._policy.config.input_features
        ):
            raise KeyError("Missing observation.state")

    def _prepare_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._validate_observation(observation)
        prepared_observation = {
            key: value
            for key, value in observation.items()
            if not key.startswith(_IMAGE_KEY_PREFIX) or key in self._expected_image_keys
        }
        for image_key in self._expected_image_keys:
            image = np.asarray(prepared_observation[image_key])
            if not image.flags.writeable:
                image = image.copy()
            prepared_observation[image_key] = image
        if "observation.state" in prepared_observation:
            # Forge JointState produces float64; ACT parameters and training data
            # use float32. LeRobot's helper preserves non-image dtypes.
            prepared_observation["observation.state"] = np.asarray(
                prepared_observation["observation.state"], dtype=np.float32
            )
        return prepare_observation_for_inference(prepared_observation, self._device)

    def _inference_autocast_context(self):
        if self._device.type == "cuda" and bool(self._policy.config.use_amp):
            return torch.autocast(device_type=self._device.type)
        return nullcontext()

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras  # keys already use observation.images.<alias>
        if observation and not self.is_observation_needed():
            raise ValueError("ACT still has queued actions; do not provide a new observation")
        if observation:
            batch = self._prepare_observation(observation)
        elif self.is_observation_needed():
            raise ValueError("ACT action queue is depleted; a fresh observation is required")
        else:
            batch = {}
        with torch.inference_mode(), self._inference_autocast_context():
            if observation:
                batch = self._preprocessor(batch)
            action = self._policy.select_action(batch)
            # Update immediately after select_action succeeds. LeRobot has consumed one
            # action at this point, even if postprocessing subsequently fails.
            if self._temporal_ensemble:
                self._queued_steps_remaining = 0
            elif observation:
                self._queued_steps_remaining = max(
                    int(self._policy.config.n_action_steps) - 1,
                    0,
                )
            else:
                self._queued_steps_remaining -= 1
            action = self._postprocessor(action)
        return action_tensor_to_numpy(action)


class ACTAsyncChunkedPolicyAdapter(ACTPolicyAdapter):
    """Single-process, non-blocking ACT action chunk inference."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        expected_image_keys: set[str],
        control_hz: float = 30.0,
        actions_per_chunk: int | None = None,
        chunk_size_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            policy,
            preprocessor,
            postprocessor,
            expected_image_keys=expected_image_keys,
        )
        if self._temporal_ensemble:
            raise ValueError(
                "ACT temporal_ensemble_coeff requires inference_mode=sync; "
                "async_chunked merges overlapping timestep chunks instead"
            )
        if isinstance(control_hz, bool) or not isinstance(control_hz, (int, float)):
            raise ValueError("control_hz must be a finite positive number")
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be finite and positive, got {control_hz}")

        raw_actions_per_chunk = (
            policy.config.n_action_steps
            if actions_per_chunk is None
            else actions_per_chunk
        )
        if isinstance(raw_actions_per_chunk, bool) or not isinstance(
            raw_actions_per_chunk,
            int,
        ):
            raise ValueError("actions_per_chunk must be a positive integer")
        if raw_actions_per_chunk <= 0:
            raise ValueError(
                f"actions_per_chunk must be positive, got {raw_actions_per_chunk}"
            )
        if isinstance(chunk_size_threshold, bool) or not isinstance(
            chunk_size_threshold,
            (int, float),
        ):
            raise ValueError("chunk_size_threshold must be a finite number in [0, 1]")
        if not math.isfinite(chunk_size_threshold) or not 0 <= chunk_size_threshold <= 1:
            raise ValueError(
                "chunk_size_threshold must be between 0 and 1, "
                f"got {chunk_size_threshold}"
            )

        self._control_hz = float(control_hz)
        self._actions_per_chunk = raw_actions_per_chunk
        self._chunk_size_threshold = float(chunk_size_threshold)
        self._action_queue = _AsyncActionQueue()

        self._condition = Condition(Lock())
        self._inference_lock = Lock()
        self._pending_observation: _TimedObservation | None = None
        self._last_requested_timestep: int | None = None
        self._generation = 0
        self._error: Exception | None = None
        self._active = False
        self._shutdown = False
        self._stopping = False
        self._needs_reset = False
        self._thread: Thread | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        device: str | None = None,
        expected_image_keys: set[str] | None = None,
        torch_compile: bool = False,
        policy_config_overrides: dict[str, Any] | None = None,
        control_hz: float = 30.0,
        actions_per_chunk: int | None = None,
        chunk_size_threshold: float = 0.5,
    ) -> ACTAsyncChunkedPolicyAdapter:
        loaded = ACTPolicyAdapter.from_pretrained(
            pretrained_path,
            device=device,
            expected_image_keys=expected_image_keys,
            torch_compile=torch_compile,
            policy_config_overrides=policy_config_overrides,
        )
        return cls(
            loaded._policy,
            loaded._preprocessor,
            loaded._postprocessor,
            expected_image_keys=loaded._expected_image_keys,
            control_hz=control_hz,
            actions_per_chunk=actions_per_chunk,
            chunk_size_threshold=chunk_size_threshold,
        )

    @property
    def failed(self) -> bool:
        with self._condition:
            return self._error is not None

    def _raise_if_failed(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise SystemExit("ACT async chunk inference failed") from error

    def start(self) -> None:
        self._raise_if_failed()

        with self._condition:
            thread = self._thread
            stopping = self._stopping
        if stopping:
            if thread is not None and thread.is_alive():
                raise RuntimeError("ACT async chunk backend is still stopping")
            with self._condition:
                self._thread = None
                self._stopping = False
                self._needs_reset = True

        with self._condition:
            needs_reset = self._needs_reset
        if needs_reset:
            self.reset()

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                if self._shutdown:
                    raise RuntimeError("ACT async chunk backend is shutting down")
                self._active = True
                self._condition.notify_all()
                return
            self._shutdown = False
            self._active = True
            self._thread = Thread(
                target=self._inference_loop,
                daemon=True,
                name="ACTAsyncChunked",
            )
            self._thread.start()
        logger.info("ACT async chunk inference thread started")

    def pause(self) -> None:
        # In-flight model calls cannot be cancelled. Generation invalidation and
        # clearing under the same condition prevent their chunks from returning.
        with self._condition:
            self._active = False
            self._generation += 1
            self._pending_observation = None
            self._last_requested_timestep = None
            self._action_queue.clear()
            self._needs_reset = True
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._active = False
            self._shutdown = True
            self._stopping = True
            self._generation += 1
            self._pending_observation = None
            self._last_requested_timestep = None
            self._action_queue.clear()
            self._needs_reset = True
            thread = self._thread
            self._condition.notify_all()

        if thread is not None and thread.is_alive():
            thread.join(timeout=_JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(
                    "ACT async chunk thread did not stop within %.1fs",
                    _JOIN_TIMEOUT_S,
                )
            else:
                logger.info("ACT async chunk inference thread stopped")

        if thread is None or not thread.is_alive():
            with self._condition:
                self._thread = None
                self._stopping = False
            self.reset()

    def reset(self) -> None:
        with self._condition:
            self._generation += 1
            reset_generation = self._generation
            self._pending_observation = None
            self._last_requested_timestep = None
            self._error = None
            self._needs_reset = True
            self._action_queue.clear()
            self._condition.notify_all()

        # Serialize processor/policy reset with background inference, but do not
        # let a stuck model call block the Dora lifecycle loop indefinitely.
        if not self._inference_lock.acquire(timeout=_JOIN_TIMEOUT_S):
            raise RuntimeError(
                "ACT async chunk reset timed out after "
                f"{_JOIN_TIMEOUT_S:.1f}s waiting for in-flight inference"
            )
        try:
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()
            # Deliberately clear again in case inference completed while reset
            # was waiting to acquire the serialization lock.
            self._action_queue.clear()
        finally:
            self._inference_lock.release()
        with self._condition:
            if self._generation == reset_generation:
                self._needs_reset = False

    def is_observation_needed(self) -> bool:
        if not self._action_queue.ready_for_observation(self._chunk_size_threshold):
            return False
        timestep = self._action_queue.observation_timestep()
        with self._condition:
            # LeRobot's server filters duplicate observations for timesteps it has
            # already accepted. Avoid copying the same Dora frame repeatedly while
            # a local inference for that timestep is pending or in flight.
            return self._last_requested_timestep != timestep

    def _copy_observation(
        self,
        observation: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        self._validate_observation(observation)
        copied: dict[str, np.ndarray] = {
            key: value
            for key, value in observation.items()
            if not key.startswith(_IMAGE_KEY_PREFIX) or key in self._expected_image_keys
        }
        if "observation.state" in copied:
            copied["observation.state"] = np.array(
                copied["observation.state"],
                dtype=np.float32,
                order="C",
                copy=True,
            )
        for image_key in self._expected_image_keys:
            copied[image_key] = np.array(
                copied[image_key],
                order="C",
                copy=True,
            )
        return copied

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray | None:
        del alias_for_cameras
        self.start()
        if observation and self.is_observation_needed():
            self._submit_observation(self._copy_observation(observation))
        self._raise_if_failed()
        return self._action_queue.get_nowait()

    def _submit_observation(self, observation: dict[str, np.ndarray]) -> None:
        timestep = self._action_queue.observation_timestep()
        with self._condition:
            if not self._active or self._shutdown:
                return
            if self._last_requested_timestep == timestep:
                return
            self._last_requested_timestep = timestep
            self._pending_observation = _TimedObservation(
                timestamp=time.time(),
                timestep=timestep,
                generation=self._generation,
                observation=observation,
            )
            self._condition.notify_all()

    def _inference_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._shutdown
                        or (self._active and self._pending_observation is not None)
                    )
                    if self._shutdown:
                        return
                    if not self._active or self._pending_observation is None:
                        continue
                    observation = self._pending_observation
                    self._pending_observation = None
                try:
                    self._predict_and_merge(observation)
                except Exception as exc:  # noqa: BLE001 - worker error boundary
                    if self._record_failure(exc, observation.generation):
                        return
                    logger.info("Ignoring ACT async error from an invalidated generation")
        except Exception as exc:  # noqa: BLE001 - final worker error boundary
            self._record_failure(exc, None)

    def _record_failure(self, error: Exception, generation: int | None) -> bool:
        with self._condition:
            if self._shutdown:
                return False
            if generation is not None and generation != self._generation:
                return False
            self._error = error
            self._active = False
            self._action_queue.clear()
        logger.error(
            "ACT async chunk inference failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        return True

    def _predict_and_merge(self, observation: _TimedObservation) -> None:
        with self._inference_lock:
            batch = self._prepare_observation(observation.observation)
            with torch.inference_mode(), self._inference_autocast_context():
                batch = self._preprocessor(batch)
                action_chunk = self._policy.predict_action_chunk(batch)
                if action_chunk.ndim == 2:
                    action_chunk = action_chunk.unsqueeze(0)
                if action_chunk.ndim != 3 or action_chunk.shape[0] != 1:
                    raise ValueError(
                        "ACT predict_action_chunk must return shape [1, steps, action_dim], "
                        f"got {tuple(action_chunk.shape)}"
                    )
                action_chunk = action_chunk[:, : self._actions_per_chunk, :]
                if action_chunk.shape[1] == 0:
                    raise ValueError("ACT predict_action_chunk returned an empty action chunk")
                processed_actions = [
                    action_tensor_to_numpy(
                        self._postprocessor(action_chunk[:, index, :])
                    ).copy()
                    for index in range(action_chunk.shape[1])
                ]

        timed_actions = [
            _TimedAction(
                timestamp=observation.timestamp + index / self._control_hz,
                timestep=observation.timestep + index,
                action=action,
            )
            for index, action in enumerate(processed_actions)
        ]
        with self._condition:
            if (
                observation.generation != self._generation
                or not self._active
                or self._shutdown
            ):
                return
            self._action_queue.merge(timed_actions)
