"""LeRobot PI0.5 policy adapter for synchronous Dora inference."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RelativeActionsProcessorStep
from lerobot_inference.inference.observation import action_tensor_to_numpy
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.loader import load_policy_bundle

from .assets import ensure_pi05_offline_assets, load_local_tokenizer
from .compatibility import (
    PI05_COMPATIBILITY_LEROBOT_044,
    apply_pi05_compatibility,
    prepare_observation_legacy_044,
)
from .loading import load_pi05_policy_strict

logger = logging.getLogger(__name__)



class PI05PolicyAdapter(LerobotPolicyAdapter):
    """PI0.5 inference through LeRobot select_action and processor pipelines."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        *,
        instruction: str = "",
        expected_image_keys: set[str],
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._instruction = str(instruction)
        self._expected_image_keys = expected_image_keys
        self._compatibility_mode = getattr(
            policy,
            "_lerobot_inference_compatibility_mode",
            "native",
        )
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
        tokenizer_path: str,
        device: str | None = None,
        instruction: str = "",
        expected_image_keys: set[str] | None = None,
        policy_config_overrides: dict[str, Any] | None = None,
        compatibility_mode: str | None = None,
        allow_rtc: bool = False,
    ) -> PI05PolicyAdapter:
        path, local_tokenizer_path = ensure_pi05_offline_assets(
            pretrained_path,
            tokenizer_path,
        )
        tokenizer = load_local_tokenizer(local_tokenizer_path)
        policy, preprocessor, postprocessor, config = load_policy_bundle(
            path,
            device=device,
            preprocessor_overrides={
                "tokenizer_processor": {
                    "tokenizer_name": None,
                    "tokenizer": tokenizer,
                }
            },
            policy_config_overrides=policy_config_overrides,
            policy_loader=load_pi05_policy_strict,
        )
        if config.type != "pi05":
            raise ValueError(f"PI05PolicyAdapter expects type pi05, got {config.type!r}")
        apply_pi05_compatibility(policy, preprocessor, compatibility_mode)
        rtc_config = getattr(config, "rtc_config", None)
        if (
            not allow_rtc
            and rtc_config is not None
            and bool(getattr(rtc_config, "enabled", False))
        ):
            raise ValueError(
                "PI0.5 RTC requires inference_mode=async_rtc; "
                "the synchronous adapter uses select_action"
            )

        synthetic_image_keys = {
            f"observation.images.empty_camera_{index}"
            for index in range(int(getattr(config, "empty_cameras", 0)))
        }
        model_image_keys = set(config.image_features) - synthetic_image_keys
        if expected_image_keys is not None and expected_image_keys != model_image_keys:
            missing = sorted(model_image_keys - expected_image_keys)
            unexpected = sorted(expected_image_keys - model_image_keys)
            raise ValueError(
                "PI0.5 camera mapping does not match checkpoint features: "
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

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        new_instruction = str(value)
        if new_instruction == self._instruction:
            return
        self._instruction = new_instruction
        self.reset()

    def validate_io_dimensions(self, state_dim: int, action_dim: int) -> None:
        state_feat = self._policy.config.input_features.get("observation.state")
        action_feat = self._policy.config.output_features.get("action")
        if state_feat is None or action_feat is None:
            raise ValueError("PI0.5 config missing observation.state or action features")
        model_state_dim = int(state_feat.shape[0])
        model_action_dim = int(action_feat.shape[0])
        if model_state_dim != state_dim or model_action_dim != action_dim:
            raise ValueError(
                "PI0.5 runtime dimensions do not match checkpoint: "
                f"state_joints={state_dim}, model_state_dim={model_state_dim}, "
                f"action_joints={action_dim}, model_action_dim={model_action_dim}"
            )

    def configure_joint_names(
        self,
        state_joint_names: list[str],
        action_joint_names: list[str],
    ) -> None:
        relative_step = next(
            (
                step
                for step in self._preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        if relative_step is None:
            return
        if state_joint_names != action_joint_names:
            raise ValueError(
                "PI0.5 relative actions require state_joints and action joints "
                "to have identical names and order"
            )
        if not action_joint_names:
            raise ValueError("PI0.5 relative actions require non-empty runtime joint names")
        configured_names = getattr(self._policy.config, "action_feature_names", None)
        if configured_names is not None and list(configured_names) != action_joint_names:
            raise ValueError(
                "Runtime action joints do not match checkpoint action_feature_names: "
                f"runtime={action_joint_names}, checkpoint={list(configured_names)}"
            )
        processor_names = relative_step.action_names
        if processor_names is not None and list(processor_names) != action_joint_names:
            raise ValueError(
                "Runtime action joints do not match processor action_names: "
                f"runtime={action_joint_names}, processor={list(processor_names)}"
            )
        relative_step.action_names = list(action_joint_names)

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
        copied = self._copy_observation(observation)
        if self._compatibility_mode == PI05_COMPATIBILITY_LEROBOT_044:
            return prepare_observation_legacy_044(
                copied,
                self._device,
                task=self._instruction,
                robot_type="",
            )
        return prepare_observation_for_inference(
            copied,
            self._device,
            task=self._instruction,
            robot_type="",
        )

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray:
        del alias_for_cameras
        if observation and not self.is_observation_needed():
            raise ValueError("PI0.5 still has queued actions; do not provide a new observation")
        if observation:
            batch = self._prepare_observation(observation)
        elif self.is_observation_needed():
            raise ValueError("PI0.5 action queue is depleted; a fresh observation is required")
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
            if observation:
                self._queued_steps_remaining = max(
                    int(self._policy.config.n_action_steps) - 1,
                    0,
                )
            else:
                self._queued_steps_remaining -= 1
            action = self._postprocessor(action)
        return action_tensor_to_numpy(action)
