"""LeRobot VLA-JEPA policy adapter for Dora inference."""

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


class VLAJEPAAdapter(LerobotPolicyAdapter):
    """Instruction-conditioned VLA-JEPA inference via LeRobot select_action.

    VLA-JEPA is a VLA combining a Qwen3-VL backbone with a flow-matching DiT
    action head (the V-JEPA2 world model is training-only and never executed
    here). Unlike ACT, the model is conditioned on a language instruction: the
    runtime instruction is injected into every batch as ``task`` and rendered
    into the Qwen prompt template by the policy itself.

    Precision is managed inside the native model (Qwen runs under bf16
    autocast, the DiT action head in float32), so unlike ACT/PI0.5 this
    adapter needs no explicit autocast context.
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
        instruction: str = "",
        expected_image_keys: set[str] | None = None,
        policy_config_overrides: dict[str, Any] | None = None,
    ) -> VLAJEPAAdapter:
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
            policy_config_overrides=policy_config_overrides,
        )
        if config.type != "vla_jepa":
            raise ValueError(f"VLAJEPAAdapter expects type vla_jepa, got {config.type!r}")
        model_image_keys = set(config.image_features)
        if expected_image_keys is not None and expected_image_keys != model_image_keys:
            missing = sorted(model_image_keys - expected_image_keys)
            unexpected = sorted(expected_image_keys - model_image_keys)
            raise ValueError(
                "VLA-JEPA camera mapping does not match checkpoint features: "
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
        """Validate observation and command dimensions independently."""
        state_feat = self._policy.config.input_features.get("observation.state")
        action_feat = self._policy.config.output_features.get("action")
        if state_feat is None or action_feat is None:
            raise ValueError("VLA-JEPA config missing observation.state or action features")
        model_state_dim = int(state_feat.shape[0])
        model_action_dim = int(action_feat.shape[0])
        if model_state_dim != state_dim or model_action_dim != action_dim:
            raise ValueError(
                "VLA-JEPA runtime dimensions do not match checkpoint: "
                f"state_joints={state_dim}, model_state_dim={model_state_dim}, "
                f"action_joints={action_dim}, model_action_dim={model_action_dim}"
            )

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        # VLA-JEPA renders the instruction into the Qwen prompt on every
        # prediction; a changed task must not reuse actions predicted for the
        # previous instruction, so the action queue is invalidated on update.
        new_instruction = str(value)
        if new_instruction == self._instruction:
            return
        self._instruction = new_instruction
        self.reset()

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._queued_steps_remaining = 0

    def pause(self) -> None:
        """Suspend execution: discard the current action chunk and queues."""
        self.reset()

    def stop(self) -> None:
        """Stop execution: same teardown semantics as pause."""
        self.reset()

    def is_observation_needed(self) -> bool:
        # VLA-JEPA predicts a chunk of chunk_size actions per observation and
        # LeRobot's select_action queue serves one step per call.
        return self._queued_steps_remaining == 0

    def _validate_observation(self, observation: dict[str, Any]) -> None:
        missing = [key for key in sorted(self._expected_image_keys) if key not in observation]
        if missing:
            raise KeyError(f"Missing camera observations: {missing}")
        if "observation.state" not in observation:
            raise KeyError("Missing observation.state")

    def _copy_observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """Validate and copy an observation into writable C-contiguous arrays."""
        self._validate_observation(observation)
        prepared = dict(observation)
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
            task=self._instruction,
            robot_type="",
        )

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras  # keys already use observation.images.<alias>
        if observation and not self.is_observation_needed():
            raise ValueError("VLA-JEPA still has queued actions; do not provide a new observation")
        if observation:
            batch = self._prepare_observation(observation)
        elif self.is_observation_needed():
            raise ValueError("VLA-JEPA action queue is depleted; a fresh observation is required")
        else:
            batch = {}

        with torch.inference_mode():
            if observation:
                batch = self._preprocessor(batch)
            action = self._policy.select_action(batch)
            # Update immediately after select_action succeeds. LeRobot has consumed one
            # action at this point, even if postprocessing subsequently fails.
            if observation:
                self._queued_steps_remaining = max(
                    int(self._policy.config.n_action_steps) - 1,
                    0,
                )
            else:
                self._queued_steps_remaining -= 1
            action = self._postprocessor(action)
        return action_tensor_to_numpy(action)
