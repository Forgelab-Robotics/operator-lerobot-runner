"""LeRobot LingBot-VA policy adapter for Dora inference."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from lerobot_inference.inference.observation import action_tensor_to_numpy
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.loader import load_policy_bundle

logger = logging.getLogger(__name__)


class LingBotVAAdapter(LerobotPolicyAdapter):
    """LingBot-VA (video-action world model) inference via LeRobot select_action.

    LingBot-VA is an autoregressive video-action world model: it predicts future
    video latents interleaved with robot actions, and re-grounds its KV cache on
    the observed keyframes as the chunk executes. The policy owns the action
    chunk queue, the keyframe buffer and the KV cache internally, so the adapter
    forwards every observation and needs no chunk bookkeeping of its own.

    The model is conditioned on a language instruction: ``task`` is rendered into
    the batch and encoded once by the frozen UMT5 text encoder. Updating the
    instruction resets the policy so the new prompt is encoded fresh.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        instruction: str,
        expected_image_keys: set[str],
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._instruction = str(instruction)
        self._expected_image_keys = expected_image_keys
        # LingBot-VA is a pure video-action model: input_features hold only the
        # camera features (observation.state is absent unless a checkpoint adds it).
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
        instruction: str = "",
        expected_image_keys: set[str] | None = None,
        policy_config_overrides: dict[str, Any] | None = None,
    ) -> LingBotVAAdapter:
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
            policy_config_overrides=policy_config_overrides,
        )
        if config.type != "lingbot_va":
            raise ValueError(f"LingBotVAAdapter expects type lingbot_va, got {config.type!r}")
        # obs_cam_keys defines the runtime camera set (input_features may carry
        # extra visual keys used at training time).
        model_image_keys = set(config.obs_cam_keys)
        if expected_image_keys is not None and expected_image_keys != model_image_keys:
            missing = sorted(model_image_keys - expected_image_keys)
            unexpected = sorted(expected_image_keys - model_image_keys)
            raise ValueError(
                "LingBot-VA camera mapping does not match checkpoint obs_cam_keys: "
                f"missing={missing}, unexpected={unexpected}, "
                f"checkpoint={sorted(model_image_keys)}"
            )
        return cls(
            policy,
            preprocessor,
            postprocessor,
            instruction=instruction,
            expected_image_keys=model_image_keys,
        )

    def validate_io_dimensions(self, state_dim: int, action_dim: int) -> None:
        """Validate action dimension; state is optional for pure-video checkpoints."""
        action_feat = self._policy.config.output_features.get("action")
        if action_feat is None:
            raise ValueError("LingBot-VA config missing action output feature")
        model_action_dim = int(action_feat.shape[0])
        if model_action_dim != action_dim:
            raise ValueError(
                "LingBot-VA runtime dimensions do not match checkpoint: "
                f"action_joints={action_dim}, model_action_dim={model_action_dim} "
                f"(used_action_channel_ids={self._policy.config.used_action_channel_ids})"
            )
        state_feat = self._policy.config.input_features.get("observation.state")
        if state_feat is not None:
            model_state_dim = int(state_feat.shape[0])
            if model_state_dim != state_dim:
                raise ValueError(
                    "LingBot-VA runtime dimensions do not match checkpoint: "
                    f"state_joints={state_dim}, model_state_dim={model_state_dim}"
                )
        # Checkpoints without observation.state skip the state check.

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        # The prompt is encoded once (frozen UMT5); a changed task must re-encode,
        # which also discards the running chunk's KV cache and keyframe buffer.
        new_instruction = str(value)
        if new_instruction == self._instruction:
            return
        self._instruction = new_instruction
        self.reset()

    def reset(self) -> None:
        # LingBotVA.reset() clears the KV cache, action queue, keyframe buffer and
        # prompt embeds.
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def pause(self) -> None:
        """Suspend execution: discard the current chunk and streaming state."""
        self.reset()

    def stop(self) -> None:
        """Stop execution: same teardown semantics as pause."""
        self.reset()

    def is_observation_needed(self) -> bool:
        # The policy buffers every observation as a candidate keyframe and feeds
        # them back into the KV cache when the action chunk is exhausted, so
        # every step must forward the current observation.
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

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras  # keys already use observation.images.<alias>
        if not observation:
            raise ValueError("LingBot-VA requires an observation on every step")
        batch = prepare_observation_for_inference(
            self._copy_observation(observation),
            self._device,
            task=self._instruction,
        )
        with torch.inference_mode():
            batch = self._preprocessor(batch)
            action = self._policy.select_action(batch)
            action = self._postprocessor(action)
        return action_tensor_to_numpy(action)
