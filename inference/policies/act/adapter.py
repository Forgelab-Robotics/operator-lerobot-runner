"""LeRobot ACT policy adapter for Dora inference."""

from __future__ import annotations

import logging
from contextlib import nullcontext
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
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
            policy_config_overrides=policy_config_overrides,
        )
        if config.type != "act":
            raise ValueError(f"ACTPolicyAdapter expects type act, got {config.type!r}")
        policy = maybe_compile_policy(policy, enabled=torch_compile)
        model_image_keys = set(config.image_features)
        if expected_image_keys is not None and expected_image_keys != model_image_keys:
            missing = sorted(model_image_keys - expected_image_keys)
            unexpected = sorted(expected_image_keys - model_image_keys)
            raise ValueError(
                "ACT camera mapping does not match checkpoint features: "
                f"missing={missing}, unexpected={unexpected}, "
                f"checkpoint={sorted(model_image_keys)}"
            )
        return cls(
            policy,
            preprocessor,
            postprocessor,
            expected_image_keys=model_image_keys,
        )

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

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras  # keys already use observation.images.<alias>
        if observation and not self.is_observation_needed():
            raise ValueError("ACT still has queued actions; do not provide a new observation")
        if observation:
            self._validate_observation(observation)
            prepared_observation = dict(observation)
            for image_key in self._expected_image_keys:
                image = np.asarray(prepared_observation[image_key])
                if not image.flags.writeable:
                    image = image.copy()
                prepared_observation[image_key] = image
            if "observation.state" in prepared_observation:
                # Forge JointState produces float64; ACT parameters and training data
                # use float32.  LeRobot's helper preserves non-image dtypes.
                prepared_observation["observation.state"] = np.asarray(
                    prepared_observation["observation.state"], dtype=np.float32
                )
            batch = prepare_observation_for_inference(prepared_observation, self._device)
        elif self.is_observation_needed():
            raise ValueError("ACT action queue is depleted; a fresh observation is required")
        else:
            batch = {}
        autocast = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and bool(self._policy.config.use_amp)
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
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
