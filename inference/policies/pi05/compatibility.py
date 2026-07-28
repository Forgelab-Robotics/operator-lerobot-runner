"""Inference compatibility for PI0.5 checkpoints trained with older LeRobot releases."""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

logger = logging.getLogger(__name__)

PI05_COMPATIBILITY_NATIVE = "native"
PI05_COMPATIBILITY_LEROBOT_044 = "lerobot_0_4_4"
_SUPPORTED_COMPATIBILITY_MODES = frozenset(
    {PI05_COMPATIBILITY_NATIVE, PI05_COMPATIBILITY_LEROBOT_044}
)


class Legacy044PrepareStateTokenizerProcessorStep(
    Pi05PrepareStateTokenizerProcessorStep
):
    """Restore the fixed-width state prompt emitted by LeRobot 0.4.4."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")

        legacy_transition = transition.copy()
        legacy_observation = dict(observation)
        legacy_observation[OBS_STATE] = pad_vector(deepcopy(state), self.max_state_dim)
        legacy_transition[TransitionKey.OBSERVATION] = legacy_observation
        return super().__call__(legacy_transition)


def normalize_pi05_compatibility_mode(raw: str | None) -> str:
    mode = PI05_COMPATIBILITY_NATIVE if raw is None else str(raw).strip().lower()
    if mode not in _SUPPORTED_COMPATIBILITY_MODES:
        raise ValueError(
            "policy.compatibility_mode must be one of: "
            f"{', '.join(sorted(_SUPPORTED_COMPATIBILITY_MODES))}; got {raw!r}"
        )
    return mode


def prepare_observation_legacy_044(
    observation: dict[str, np.ndarray],
    device: torch.device,
    *,
    task: str = "",
    robot_type: str = "",
) -> dict[str, Any]:
    """Reproduce LeRobot 0.4.4 observation tensor conversion."""
    prepared: dict[str, Any] = {}
    for name, value in observation.items():
        tensor = torch.from_numpy(value)
        if name.startswith("observation.images"):
            tensor = tensor.to(torch.float32) / 255
            tensor = tensor.permute(2, 0, 1).contiguous()
        else:
            tensor = tensor.to(torch.float32)
        prepared[name] = tensor.unsqueeze(0).to(device)
    prepared["task"] = task
    prepared["robot_type"] = robot_type
    return prepared


def _resize_with_pad_legacy_044(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Reproduce LeRobot 0.4.4 PI0.5 letterboxing, including its float padding value."""
    channels_last = images.shape[-1] <= 4
    if channels_last:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    elif images.dim() == 3:
        images = images.unsqueeze(0)

    _, _, current_height, current_width = images.shape
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        padding_value = 0
    elif images.dtype == torch.float32:
        resized = resized.clamp(-1.0, 1.0)
        padding_value = -1.0
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded = F.pad(
        resized,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=padding_value,
    )
    if channels_last:
        padded = padded.permute(0, 2, 3, 1)
    return padded


def _legacy_044_preprocess_images(
    policy: Any,
    batch: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Reproduce PI0.5 image preparation from LeRobot 0.4.4."""
    images: list[torch.Tensor] = []
    image_masks: list[torch.Tensor] = []
    device = next(policy.parameters()).device

    present_keys = [key for key in policy.config.image_features if key in batch]
    missing_keys = [key for key in policy.config.image_features if key not in batch]
    if not present_keys:
        raise ValueError(
            "All image features are missing from the batch. At least one expected. "
            f"(batch: {batch.keys()}) (image_features: {policy.config.image_features})"
        )

    for key in present_keys:
        image = batch[key]
        if image.device != device:
            image = image.to(device)
        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        channels_first = image.shape[1] == 3
        if channels_first:
            image = image.permute(0, 2, 3, 1)
        if image.shape[1:3] != policy.config.image_resolution:
            image = _resize_with_pad_legacy_044(
                image,
                *policy.config.image_resolution,
            )
        # In 0.4.4 float letterbox pixels were -1 before this conversion, so
        # their final SigLIP value was -3. Old checkpoints were trained with it.
        image = image * 2.0 - 1.0
        if channels_first:
            image = image.permute(0, 3, 1, 2)

        images.append(image)
        image_masks.append(torch.ones(image.shape[0], dtype=torch.bool, device=device))

    reference_image = images[-1]
    reference_mask = image_masks[-1]
    for _ in missing_keys:
        images.append(torch.ones_like(reference_image) * -1)
        image_masks.append(torch.zeros_like(reference_mask))
    return images, image_masks


def _install_legacy_044_language_embedding_scale(policy: Any) -> None:
    embedding_host = policy.model.paligemma_with_expert
    original_embed = embedding_host.embed_language_tokens

    def legacy_embed_language_tokens(_self: Any, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = original_embed(tokens)
        return embeddings * math.sqrt(embeddings.shape[-1])

    embedding_host.embed_language_tokens = MethodType(
        legacy_embed_language_tokens,
        embedding_host,
    )


def _replace_state_tokenizer_step(preprocessor: PolicyProcessorPipeline) -> None:
    matches = [
        (index, step)
        for index, step in enumerate(preprocessor.steps)
        if isinstance(step, Pi05PrepareStateTokenizerProcessorStep)
    ]
    if len(matches) != 1:
        raise ValueError(
            "LeRobot 0.4.4 PI0.5 compatibility requires exactly one "
            "pi05_prepare_state_tokenizer_processor_step; "
            f"found {len(matches)}"
        )
    index, step = matches[0]
    replacement = Legacy044PrepareStateTokenizerProcessorStep(
        max_state_dim=int(step.max_state_dim),
        task_key=str(step.task_key),
    )
    steps = list(preprocessor.steps)
    steps[index] = replacement
    preprocessor.steps = steps


def apply_pi05_compatibility(
    policy: Any,
    preprocessor: PolicyProcessorPipeline,
    mode: str | None,
) -> str:
    """Apply instance-scoped compatibility behavior without modifying checkpoint files."""
    normalized_mode = normalize_pi05_compatibility_mode(mode)
    if normalized_mode == PI05_COMPATIBILITY_NATIVE:
        return normalized_mode

    existing_mode = getattr(policy, "_lerobot_inference_compatibility_mode", None)
    if existing_mode is not None:
        if existing_mode != normalized_mode:
            raise ValueError(
                f"PI0.5 policy already uses compatibility mode {existing_mode!r}"
            )
        return normalized_mode

    _replace_state_tokenizer_step(preprocessor)
    policy._preprocess_images = MethodType(_legacy_044_preprocess_images, policy)
    _install_legacy_044_language_embedding_scale(policy)
    policy._lerobot_inference_compatibility_mode = normalized_mode
    logger.warning(
        "Enabled PI0.5 LeRobot 0.4.4 checkpoint compatibility: "
        "fixed-width state tokens, legacy image letterboxing, and language embedding scaling"
    )
    return normalized_mode
