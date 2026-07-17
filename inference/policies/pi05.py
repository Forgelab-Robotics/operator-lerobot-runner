"""Background-thread Pi0.5 inference via LeRobot 0.6."""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

from lerobot_inference.common.pretrained_assets import ensure_pi05_offline_assets, set_tokenizer_path
from lerobot_inference.inference.observation import action_tensor_to_numpy, observation_to_policy_batch
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.loader import load_policy_bundle

logger = logging.getLogger(__name__)

_cv_action = threading.Condition()
_cv_observation = threading.Condition()


class PI05PolicyAdapter(LerobotPolicyAdapter):
    """Async chunk inference for Pi0.5 (LeRobot predict_action_chunk + action queue)."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        instruction: str = "",
        expected_image_keys: set[str],
        get_actions_threshold: int = 0,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._instruction = instruction
        self._expected_image_keys = expected_image_keys
        self._get_actions_threshold = get_actions_threshold

        self._action_queue: deque[np.ndarray] = deque()
        self._observation_needed = True
        self._raw_observation: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._is_running = False
        self._thread: threading.Thread | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        tokenizer_path: str,
        device: str | None = None,
        instruction: str = "",
        expected_image_keys: set[str] | None = None,
        get_actions_threshold: int = 0,
    ) -> PI05PolicyAdapter:
        set_tokenizer_path(tokenizer_path)
        ensure_pi05_offline_assets(pretrained_path, tokenizer_path)
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
        )
        if config.type != "pi05":
            raise ValueError(f"PI05PolicyAdapter expects type pi05, got {config.type!r}")
        image_keys = expected_image_keys or {
            key for key in config.input_features if key.startswith("observation.images.")
        }
        return cls(
            policy,
            preprocessor,
            postprocessor,
            instruction=instruction,
            expected_image_keys=image_keys,
            get_actions_threshold=get_actions_threshold,
        )

    def reset(self) -> None:
        self._observation_needed = True
        with _cv_action:
            self._action_queue.clear()
            model_queue = getattr(self._policy, "_action_queue", None)
            if hasattr(model_queue, "clear"):
                model_queue.clear()
            _cv_action.notify_all()
        with _cv_observation:
            self._drain_observation_queue()
            _cv_observation.notify_all()

    def _drain_observation_queue(self) -> None:
        while True:
            try:
                self._raw_observation.get_nowait()
            except queue.Empty:
                break
            try:
                self._raw_observation.task_done()
            except ValueError:
                pass

    def is_observation_needed(self) -> bool:
        return self._observation_needed

    def _validate_observation(self, observation: dict[str, Any]) -> None:
        missing = [
            key for key in sorted(self._expected_image_keys) if key not in observation
        ]
        if missing:
            raise KeyError(f"Missing camera observations: {missing}")
        if "observation.state" not in observation:
            raise KeyError("Missing observation.state")

    def _prepare_batch(self, observation: dict[str, Any]) -> dict[str, Any]:
        batch = observation_to_policy_batch(observation)
        batch["task"] = self._instruction or ""
        batch["robot_type"] = ""
        return self._preprocessor(batch)

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras
        self._ensure_thread()
        return self._consume_action(observation)

    def _ensure_thread(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        logger.info("Pi0.5 background inference thread started")

    def _consume_action(self, observation: dict[str, Any]) -> np.ndarray:
        if observation and self._observation_needed and self._raw_observation.empty():
            self._validate_observation(observation)
            with _cv_observation:
                self._raw_observation.put(observation)
                _cv_observation.notify()

        with _cv_action:
            while len(self._action_queue) == 0:
                _cv_action.wait()
            action = self._action_queue.popleft()
            _cv_action.notify_all()
        return action

    def _inference_loop(self) -> None:
        while self._is_running:
            with _cv_action:
                while len(self._action_queue) > self._get_actions_threshold:
                    _cv_action.wait()

            if len(self._action_queue) <= self._get_actions_threshold:
                self._observation_needed = True

            with _cv_observation:
                while self._raw_observation.empty():
                    _cv_observation.wait()
                if not self._is_running:
                    break
                try:
                    observation = self._raw_observation.get()
                    self._observation_needed = False
                    self._raw_observation.task_done()
                except queue.Empty:
                    continue

            self._infer_chunk(observation)

    def _infer_chunk(self, observation: dict[str, Any]) -> None:
        infer_start_len = len(self._action_queue)
        batch = self._prepare_batch(observation)
        with torch.inference_mode():
            actions = self._policy.predict_action_chunk(batch)[
                :, : self._policy.config.n_action_steps
            ]
            self._policy._action_queue.extend(actions.transpose(0, 1))

        with _cv_action:
            infer_delay = infer_start_len - len(self._action_queue)
            self._action_queue.clear()
            skipped = 0
            while len(self._policy._action_queue) > 0:
                skipped += 1
                action = self._policy._action_queue.popleft()
                if skipped <= infer_delay:
                    continue
                action = self._postprocessor(action)
                self._action_queue.append(action_tensor_to_numpy(action))
            _cv_action.notify_all()

        logger.debug("Pi0.5 chunk ready, queue size=%s", len(self._action_queue))
