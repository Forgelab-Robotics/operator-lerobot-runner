"""Optional torch.compile helpers for inference (aligned with act-local-trainer)."""

from __future__ import annotations

import logging
from typing import Any

import torch
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


def maybe_compile_policy(
    policy: PreTrainedPolicy,
    *,
    enabled: bool = True,
) -> PreTrainedPolicy:
    """Compile ACT vision backbone when enabled (ResNet IntermediateLayerGetter).

    Matches act-local-trainer ``BackboneBase`` behavior: compile the backbone,
    log success / warning, never raise on failure.
    """
    if not enabled:
        return policy

    model = getattr(policy, "model", None)
    backbone = getattr(model, "backbone", None) if model is not None else None
    if backbone is None:
        logger.debug("torch.compile skipped: no policy.model.backbone")
        return policy

    try:
        model.backbone = torch.compile(backbone)
        logger.info("Successfully compiled ACT backbone (ResNet).")
    except Exception as exc:  # noqa: BLE001 — match act-local-trainer soft-fail
        logger.warning("Could not compile backbone: %s", exc)
    return policy


def compile_enabled_from_policy_config(policy_config: dict[str, Any]) -> bool:
    """Read ``policy.torch_compile`` (default True)."""
    raw = policy_config.get("torch_compile", True)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)
