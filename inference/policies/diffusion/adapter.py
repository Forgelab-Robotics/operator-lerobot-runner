"""LeRobot Diffusion Policy adapter for Dora inference."""

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


class DiffusionPolicyAdapter(LerobotPolicyAdapter):
    """Diffusion Policy inference via LeRobot select_action + processor pipelines.

    LeRobot's DiffusionPolicy owns both queues internally: every call to
    select_action takes the current observation, stacks it with the previous
    ``n_obs_steps - 1`` frames, and re-runs the diffusion denoising loop only when
    the action chunk queue is depleted. The adapter therefore forwards every
    observation and needs no chunk bookkeeping of its own.
    """

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
        # Pure-visual diffusion configs exist; the state feature is optional.
        self._has_state = "observation.state" in policy.config.input_features
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
    ) -> DiffusionPolicyAdapter:
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
            policy_config_overrides=policy_config_overrides,
        )
        if config.type != "diffusion":
            raise ValueError(f"DiffusionPolicyAdapter expects type diffusion, got {config.type!r}")
        policy = maybe_compile_policy(policy, enabled=torch_compile)
        model_image_keys = set(config.image_features)
        if expected_image_keys is not None and expected_image_keys != model_image_keys:
            missing = sorted(model_image_keys - expected_image_keys)
            unexpected = sorted(expected_image_keys - model_image_keys)
            raise ValueError(
                "Diffusion camera mapping does not match checkpoint features: "
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
        action_feat = self._policy.config.output_features.get("action")
        if action_feat is None:
            raise ValueError("Diffusion config missing action output feature")
        model_action_dim = int(action_feat.shape[0])
        if model_action_dim != action_dim:
            raise ValueError(
                "Diffusion runtime dimensions do not match checkpoint: "
                f"action_joints={action_dim}, model_action_dim={model_action_dim}"
            )
        state_feat = self._policy.config.input_features.get("observation.state")
        if state_feat is not None:
            model_state_dim = int(state_feat.shape[0])
            if model_state_dim != state_dim:
                raise ValueError(
                    "Diffusion runtime dimensions do not match checkpoint: "
                    f"state_joints={state_dim}, model_state_dim={model_state_dim}"
                )
        # Configs without observation.state (pure-visual policies) skip the state check.

    def reset(self) -> None:
        # DiffusionPolicy.reset() clears its observation-history and action queues.
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def pause(self) -> None:
        """Suspend execution: discard the internal observation history and queues."""
        self.reset()

    def stop(self) -> None:
        """Stop execution: same teardown semantics as pause."""
        self.reset()

    def is_observation_needed(self) -> bool:
        # The policy stacks the last n_obs_steps observations itself and re-plans
        # only when its action chunk is exhausted, so every step needs a frame.
        return True

    def _validate_observation(self, observation: dict[str, Any]) -> None:
        missing = [key for key in sorted(self._expected_image_keys) if key not in observation]
        if missing:
            raise KeyError(f"Missing camera observations: {missing}")
        if self._has_state and "observation.state" not in observation:
            raise KeyError("Missing observation.state")

    def _copy_observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """Validate and copy an observation into writable C-contiguous arrays."""
        self._validate_observation(observation)
        prepared = dict(observation)
        if "observation.state" in prepared:
            state = np.array(
                prepared["observation.state"],
                dtype=np.float32,
                order="C",
                copy=True,
            )
            if state.ndim != 1:
                raise ValueError(f"Expected 1-D observation.state, got shape={state.shape}")
            if not np.isfinite(state).all():
                raise ValueError("observation.state must contain only finite values")
            prepared["observation.state"] = state

        for image_key in self._expected_image_keys:
            image = np.asarray(prepared[image_key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Expected HWC RGB image for {image_key}, got shape={image.shape}"
                )
            if image.dtype == np.uint8:
                pass
            elif np.issubdtype(image.dtype, np.floating):
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError(
                        f"Floating image {image_key} must contain finite values in [0, 1]"
                    )
            else:
                raise ValueError(
                    f"Image {image_key} must use uint8 or floating [0, 1] values, "
                    f"got dtype={image.dtype}"
                )
            prepared[image_key] = np.array(image, order="C", copy=True)
        return prepared

    def _prepare_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return prepare_observation_for_inference(
            self._copy_observation(observation),
            self._device,
        )

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras  # keys already use observation.images.<alias>
        if not observation:
            raise ValueError("Diffusion policy requires an observation on every step")
        batch = self._prepare_observation(observation)
        autocast = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and bool(self._policy.config.use_amp)
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            batch = self._preprocessor(batch)
            action = self._policy.select_action(batch)
            action = self._postprocessor(action)
        return action_tensor_to_numpy(action)
