"""LeRobot ACT policy adapter for Dora inference."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

from lerobot_inference.inference.compile_utils import maybe_compile_policy
from lerobot_inference.inference.observation import action_tensor_to_numpy, observation_to_policy_batch
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.loader import load_policy_bundle

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        device: str | None = None,
        expected_image_keys: set[str] | None = None,
        torch_compile: bool = True,
    ) -> ACTPolicyAdapter:
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
        )
        if config.type != "act":
            raise ValueError(f"ACTPolicyAdapter expects type act, got {config.type!r}")
        policy = maybe_compile_policy(policy, enabled=torch_compile)
        image_keys = expected_image_keys or {
            key for key in config.input_features if key.startswith("observation.images.")
        }
        return cls(
            policy,
            preprocessor,
            postprocessor,
            expected_image_keys=image_keys,
        )

    def validate_joint_count(self, joint_count: int) -> None:
        """Require joints length to match model state/action dims (pick_and_place habit)."""
        state_feat = self._policy.config.input_features.get("observation.state")
        action_feat = self._policy.config.output_features.get("action")
        if state_feat is None or action_feat is None:
            raise ValueError("ACT config missing observation.state or action features")
        state_dim = int(state_feat.shape[0])
        action_dim = int(action_feat.shape[0])
        if state_dim != joint_count or action_dim != joint_count:
            raise ValueError(
                f"joints 数量={joint_count} 必须等于模型 state_dim={state_dim} "
                f"与 action_dim={action_dim}（与 pick_and_place 一致：配置与机器人/模型对齐）"
            )

    def reset(self) -> None:
        self._policy.reset()

    def is_observation_needed(self) -> bool:
        queue = getattr(self._policy, "_action_queue", None)
        if queue is not None and hasattr(queue, "__len__"):
            return len(queue) == 0
        if getattr(self._policy.config, "temporal_ensemble_coeff", None) is not None:
            return True
        return True

    def _validate_observation(self, observation: dict[str, Any]) -> None:
        missing = [
            key
            for key in sorted(self._expected_image_keys)
            if key not in observation
        ]
        if missing:
            raise KeyError(f"Missing camera observations: {missing}")
        if "observation.state" not in observation and any(
            key == "observation.state" for key in self._policy.config.input_features
        ):
            raise KeyError("Missing observation.state")

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras  # keys already use observation.images.<alias>
        # ``forge_policy`` only provides an observation when
        # ``is_observation_needed`` returns True.  LeRobot ACT keeps the remaining
        # chunk actions in ``_action_queue``, so the intermediate ticks correctly
        # arrive here as an empty dict.  Requiring images on those ticks made every
        # action after the first one fail instead of consuming the queued action.
        if observation:
            self._validate_observation(observation)
            batch = observation_to_policy_batch(observation)
            batch = self._preprocessor(batch)
        elif self.is_observation_needed():
            raise ValueError(
                "ACT needs a fresh observation because its action queue is empty"
            )
        else:
            batch = {}
        with torch.inference_mode():
            action = self._policy.select_action(batch)
        action = self._postprocessor(action)
        return action_tensor_to_numpy(action)
