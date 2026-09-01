"""LeRobot SmolVLA policy adapter for Dora inference."""

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


class SmolVLAAdapter(LerobotPolicyAdapter):
    """Instruction-conditioned SmolVLA inference via LeRobot select_action.

    SmolVLA pairs a SmolVLM2 vision-language backbone with a flow-matching
    action expert. Like VLA-JEPA it is language-conditioned: the runtime
    instruction is injected into every batch as ``task``. Unlike ACT it emits
    an action chunk of ``n_action_steps`` per inference, and LeRobot's
    ``select_action`` queue serves one step per call.

    Two checkpoint-specific traps this adapter guards against:

    * Released RoboCasa/LIBERO SmolVLA checkpoints declare a **wrong**
      ``observation.state`` shape in ``config.json`` (e.g. ``[6]`` while the
      model requires 16). ``validate_io_dimensions`` therefore compares against
      the normalization statistics shipped with the checkpoint rather than the
      declared feature shape, and reports the mismatch explicitly instead of
      failing later with an opaque tensor-size error.
    * The backbone (``HuggingFaceTB/SmolVLM2-*``) is fetched by name at load
      time. For offline deployment point ``vlm_model_name`` at a local
      directory, otherwise loading reaches out to the Hub.
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
    ) -> SmolVLAAdapter:
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            pretrained_path,
            device=device,
            policy_config_overrides=policy_config_overrides,
        )
        if config.type != "smolvla":
            raise ValueError(f"SmolVLAAdapter expects type smolvla, got {config.type!r}")
        model_image_keys = set(config.image_features)
        if expected_image_keys is not None and expected_image_keys != model_image_keys:
            missing = sorted(model_image_keys - expected_image_keys)
            unexpected = sorted(expected_image_keys - model_image_keys)
            raise ValueError(
                "SmolVLA camera mapping does not match checkpoint features: "
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

    def _checkpoint_state_dim(self) -> int | None:
        """State width the model actually consumes.

        Prefers the normalization buffer over ``config.input_features`` because
        several released SmolVLA checkpoints declare the wrong shape there.
        """
        for module in self._preprocessor.steps if hasattr(self._preprocessor, "steps") else []:
            stats = getattr(module, "stats", None) or getattr(module, "_stats", None)
            if isinstance(stats, dict):
                entry = stats.get("observation.state")
                if isinstance(entry, dict):
                    for key in ("mean", "min", "max", "std"):
                        value = entry.get(key)
                        if value is not None:
                            return int(np.asarray(getattr(value, "cpu", lambda: value)()).reshape(-1).shape[0])
        feature = self._policy.config.input_features.get("observation.state")
        return int(feature.shape[0]) if feature is not None else None

    def validate_io_dimensions(self, state_dim: int, action_dim: int) -> None:
        action_feat = self._policy.config.output_features.get("action")
        if action_feat is None:
            raise ValueError("SmolVLA config missing action feature")
        model_action_dim = int(action_feat.shape[0])
        if model_action_dim != action_dim:
            raise ValueError(
                "SmolVLA runtime action dimension does not match checkpoint: "
                f"action_joints={action_dim}, model_action_dim={model_action_dim}"
            )

        model_state_dim = self._checkpoint_state_dim()
        declared = self._policy.config.input_features.get("observation.state")
        declared_dim = int(declared.shape[0]) if declared is not None else None
        if model_state_dim is not None and declared_dim is not None and model_state_dim != declared_dim:
            logger.warning(
                "SmolVLA checkpoint declares observation.state=%d but its normalization "
                "statistics are %d-dimensional; trusting the statistics. Feed %d-D state.",
                declared_dim,
                model_state_dim,
                model_state_dim,
            )
        if model_state_dim is not None and model_state_dim != state_dim:
            raise ValueError(
                "SmolVLA runtime state dimension does not match checkpoint: "
                f"state_joints={state_dim}, model_state_dim={model_state_dim}"
            )

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        # The instruction is rendered into the VLM prompt on every prediction;
        # a changed task must not reuse actions predicted for the previous one.
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
        self.reset()

    def stop(self) -> None:
        self.reset()

    def is_observation_needed(self) -> bool:
        return self._queued_steps_remaining == 0

    def _validate_observation(self, observation: dict[str, Any]) -> None:
        missing = [key for key in sorted(self._expected_image_keys) if key not in observation]
        if missing:
            raise KeyError(f"Missing camera observations: {missing}")
        if "observation.state" not in observation:
            raise KeyError("Missing observation.state")

    def _copy_observation(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        self._validate_observation(observation)
        prepared = dict(observation)
        state = np.array(prepared["observation.state"], dtype=np.float32, order="C", copy=True)
        if state.ndim != 1:
            raise ValueError(f"Expected 1-D observation.state, got shape={state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("observation.state must contain only finite values")
        prepared["observation.state"] = state

        for image_key in self._expected_image_keys:
            image = np.asarray(prepared[image_key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image for {image_key}, got shape={image.shape}")
            if image.dtype == np.uint8:
                pass
            elif np.issubdtype(image.dtype, np.floating):
                if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError(f"Floating image {image_key} must contain finite values in [0, 1]")
            else:
                raise ValueError(
                    f"Image {image_key} must use uint8 or floating [0, 1] values, got dtype={image.dtype}"
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
            raise ValueError("SmolVLA still has queued actions; do not provide a new observation")
        if observation:
            batch = self._prepare_observation(observation)
        elif self.is_observation_needed():
            raise ValueError("SmolVLA action queue is depleted; a fresh observation is required")
        else:
            batch = {}

        with torch.inference_mode():
            if observation:
                batch = self._preprocessor(batch)
            action = self._policy.select_action(batch)
            # Update immediately after select_action succeeds: LeRobot has
            # consumed one action even if postprocessing subsequently fails.
            if observation:
                self._queued_steps_remaining = max(int(self._policy.config.n_action_steps) - 1, 0)
            else:
                self._queued_steps_remaining -= 1
            action = self._postprocessor(action)
        return action_tensor_to_numpy(action)
