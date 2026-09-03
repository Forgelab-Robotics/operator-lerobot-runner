"""LeRobot PI0.5 policy adapter for synchronous Dora inference."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from types import MethodType
from typing import Any

import numpy as np
import torch
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RelativeActionsProcessorStep
from lerobot_inference.inference.observation import action_tensor_to_numpy
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.loader import load_policy_bundle

from .assets import ensure_pi05_offline_assets, load_local_tokenizer
from .loading import load_pi05_policy_strict

logger = logging.getLogger(__name__)

_IMAGE_KEY_PREFIX = "observation.images."
_POSITION_FEATURE_SUFFIX = ".pos"


def _position_feature_joint_names(names: Sequence[str]) -> list[str]:
    """Map LeRobot position feature labels to Forge canonical joint names."""
    return [str(name).removesuffix(_POSITION_FEATURE_SUFFIX) for name in names]


def _preprocess_images_in_checkpoint_order(
    policy: PreTrainedPolicy,
    batch: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Prepare PI0.5 images without compacting cameras into earlier slots."""
    device = next(policy.parameters()).device
    processed: dict[str, torch.Tensor] = {}

    for key in policy.config.image_features:
        if key not in batch:
            continue
        image = batch[key].to(device=device, dtype=torch.float32)
        channels_first = image.shape[1] == 3
        if channels_first:
            image = image.permute(0, 2, 3, 1)
        if image.shape[1:3] != policy.config.image_resolution:
            image = resize_with_pad_torch(image, *policy.config.image_resolution)
        image = image * 2.0 - 1.0
        if channels_first:
            image = image.permute(0, 3, 1, 2)
        processed[key] = image

    if not processed:
        raise ValueError(
            "All image features are missing from the batch. At least one expected. "
            f"(batch: {batch.keys()}) (image_features: {policy.config.image_features})"
        )

    template = next(iter(processed.values()))
    batch_size = template.shape[0]
    images: list[torch.Tensor] = []
    image_masks: list[torch.Tensor] = []
    for key in policy.config.image_features:
        image = processed.get(key)
        if image is None:
            images.append(torch.full_like(template, -1))
            image_masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
        else:
            images.append(image)
            image_masks.append(torch.ones(batch_size, dtype=torch.bool, device=device))
    return images, image_masks


def _preserve_pi05_camera_slots(policy: PreTrainedPolicy) -> None:
    policy._preprocess_images = MethodType(  # type: ignore[attr-defined]
        _preprocess_images_in_checkpoint_order,
        policy,
    )


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
        runtime_image_keys = (
            set(expected_image_keys)
            if expected_image_keys is not None
            else model_image_keys
        )
        active_image_keys = model_image_keys & runtime_image_keys
        missing = sorted(model_image_keys - runtime_image_keys)
        unexpected = sorted(runtime_image_keys - model_image_keys)
        if not active_image_keys and model_image_keys:
            raise ValueError(
                "PI0.5 runtime image inputs have no keys in common with the checkpoint: "
                f"runtime={sorted(runtime_image_keys)}, checkpoint={sorted(model_image_keys)}"
            )
        if missing:
            logger.warning(
                "PI0.5 will mask checkpoint image features not provided at runtime: %s",
                missing,
            )
            _preserve_pi05_camera_slots(policy)
        if unexpected:
            logger.warning(
                "Ignoring image inputs not used by the PI0.5 checkpoint: %s",
                unexpected,
            )
        return cls(
            policy,
            preprocessor,
            postprocessor,
            instruction=instruction,
            expected_image_keys=active_image_keys,
        )

    @property
    def required_image_keys(self) -> frozenset[str]:
        return frozenset(self._expected_image_keys)

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
        runtime_feature_names = _position_feature_joint_names(action_joint_names)
        configured_names = getattr(self._policy.config, "action_feature_names", None)
        if configured_names is not None and (
            _position_feature_joint_names(configured_names) != runtime_feature_names
        ):
            raise ValueError(
                "Runtime action joints do not match checkpoint action_feature_names: "
                f"runtime={action_joint_names}, checkpoint={list(configured_names)}"
            )
        processor_names = relative_step.action_names
        if processor_names is not None and (
            _position_feature_joint_names(processor_names) != runtime_feature_names
        ):
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
        prepared = {
            key: value
            for key, value in observation.items()
            if not key.startswith(_IMAGE_KEY_PREFIX)
            or key in self._expected_image_keys
        }
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

    def _inference_autocast_context(self):
        if self._device.type != "cuda" or not bool(self._policy.config.use_amp):
            return nullcontext()

        configured_dtype = getattr(self._policy.config, "dtype", None)
        if isinstance(configured_dtype, torch.dtype):
            model_dtype = configured_dtype
        else:
            dtype_name = str(configured_dtype).removeprefix("torch.").lower()
            dtype_by_name = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            model_dtype = dtype_by_name.get(dtype_name)

        if model_dtype is None:
            raise ValueError(
                "PI0.5 use_amp requires model dtype to be bfloat16, float16, or float32; "
                f"got {configured_dtype!r}"
            )
        if model_dtype == torch.float32:
            # CUDA autocast does not support float32 as its target dtype. Keeping the
            # context disabled preserves the model's configured precision.
            return nullcontext()
        return torch.autocast(device_type=self._device.type, dtype=model_dtype)

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

        with torch.inference_mode(), self._inference_autocast_context():
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
