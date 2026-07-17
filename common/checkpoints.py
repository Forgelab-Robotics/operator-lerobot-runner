"""Checkpoint directory conventions and alias publishing (self-contained)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR_NAME = "checkpoints"
CHECKPOINT_LAST_ALIAS = "checkpoint_last.pt"
CHECKPOINT_ALIAS_PREFIX = "checkpoint_"
MODEL_WEIGHTS_NAME = "model.safetensors"
PRETRAINED_MODEL_DIR = "pretrained_model"


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def publish_checkpoint_aliases(checkpoint_dir: Path | str) -> list[Path]:
    """Expose flat checkpoints/checkpoint_*.pt symlinks for downstream tooling."""
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_root = checkpoint_dir.parent
    weights = checkpoint_dir / PRETRAINED_MODEL_DIR / MODEL_WEIGHTS_NAME
    if not weights.is_file():
        logger.warning("Checkpoint weights not found, skipping alias: %s", weights)
        return []

    rel_weights = weights.relative_to(ckpt_root)
    step_alias = ckpt_root / f"{CHECKPOINT_ALIAS_PREFIX}{checkpoint_dir.name}.pt"
    last_alias = ckpt_root / CHECKPOINT_LAST_ALIAS
    _replace_symlink(step_alias, rel_weights)
    _replace_symlink(last_alias, rel_weights)
    logger.info("Published checkpoint aliases: %s, %s", step_alias.name, last_alias.name)
    return [step_alias, last_alias]
