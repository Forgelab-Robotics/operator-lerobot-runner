"""Strict local weight loading for LeRobot PI0.5 checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def load_pi05_policy_strict(
    policy_cls: type[PreTrainedPolicy],
    path: Path,
    config: PreTrainedConfig,
) -> PreTrainedPolicy:
    """Load local PI0.5 weights without LeRobot's exception-swallowing fallback."""
    n_action_steps = getattr(config, "n_action_steps", None)
    chunk_size = getattr(config, "chunk_size", None)
    if (
        isinstance(n_action_steps, bool)
        or isinstance(chunk_size, bool)
        or not isinstance(n_action_steps, int)
        or not isinstance(chunk_size, int)
        or not 1 <= n_action_steps <= chunk_size
    ):
        raise ValueError(
            "PI0.5 requires integer action chunk dimensions with "
            f"1 <= n_action_steps <= chunk_size, got {n_action_steps=} and {chunk_size=}"
        )

    policy = policy_cls(config)
    weights_path = path / "model.safetensors"
    try:
        state_dict = load_file(str(weights_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load PI0.5 weights from {weights_path}") from exc
    if not state_dict:
        raise RuntimeError(f"PI0.5 weights are empty: {weights_path}")

    try:
        fixed_state_dict = policy._fix_pytorch_state_dict_keys(state_dict, config)
        remapped_state_dict: dict[str, torch.Tensor] = {}
        remapped_sources: dict[str, str] = {}
        for source_key, value in fixed_state_dict.items():
            target_key = (
                source_key if source_key.startswith("model.") else f"model.{source_key}"
            )
            if target_key in remapped_state_dict:
                raise ValueError(
                    "Ambiguous PI0.5 weight keys map to the same parameter "
                    f"{target_key!r}: {remapped_sources[target_key]!r} and {source_key!r}"
                )
            remapped_state_dict[target_key] = value
            remapped_sources[target_key] = source_key
        policy.load_state_dict(remapped_state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(f"PI0.5 weights are incompatible with config in {path}") from exc

    logger.info("Strictly loaded PI0.5 weights from %s", weights_path)
    return policy
