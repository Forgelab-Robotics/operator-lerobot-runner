"""Resolve lerobot_trainer run directories to pretrained_model paths."""

from __future__ import annotations

import json
from pathlib import Path

from lerobot_inference.common.checkpoints import (
    CHECKPOINT_LAST_ALIAS,
    CHECKPOINTS_DIR_NAME,
    MODEL_WEIGHTS_NAME,
    PRETRAINED_MODEL_DIR,
)

_RUN_ARTIFACT_NAMES = ("training_config.yaml", "summary.json", "metrics.json")


def _read_summary_latest(run_dir: Path) -> str | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = data.get("artifacts") or {}
    latest = artifacts.get("latest_checkpoint")
    return str(latest) if latest else None


def _resolve_checkpoint_alias(run_dir: Path, checkpoint: str) -> Path:
    ckpt_root = run_dir / CHECKPOINTS_DIR_NAME
    alias = ckpt_root / checkpoint
    if alias.is_symlink():
        target = (ckpt_root / alias.readlink()).resolve()
        if target.is_file():
            return target.parent if target.name == MODEL_WEIGHTS_NAME else target
    if alias.is_file():
        if alias.name == MODEL_WEIGHTS_NAME:
            return alias.parent
        if alias.suffix == ".pt":
            return alias.resolve().parent
    raise FileNotFoundError(f"Checkpoint alias not found under {ckpt_root}: {checkpoint}")


def _resolve_step_dir(ckpt_root: Path, step_name: str) -> Path:
    step_dir = ckpt_root / step_name
    pretrained = step_dir / PRETRAINED_MODEL_DIR
    weights = pretrained / MODEL_WEIGHTS_NAME
    if weights.is_file():
        return pretrained
    raise FileNotFoundError(f"pretrained_model not found for step {step_name}: {weights}")


def resolve_pretrained_path(
    *,
    run_dir: str | Path | None = None,
    pretrained_path: str | Path | None = None,
    checkpoint: str = "last",
) -> Path:
    """Resolve a lerobot_trainer artifact to a pretrained_model directory."""
    if pretrained_path is not None:
        path = Path(pretrained_path).expanduser().resolve()
        if (path / MODEL_WEIGHTS_NAME).is_file():
            return path
        nested = path / PRETRAINED_MODEL_DIR / MODEL_WEIGHTS_NAME
        if nested.is_file():
            return nested.parent
        raise FileNotFoundError(
            f"pretrained_path must contain {MODEL_WEIGHTS_NAME}: {path}"
        )

    if run_dir is None:
        raise ValueError("Either run_dir or pretrained_path is required.")

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    direct = run_dir / PRETRAINED_MODEL_DIR / MODEL_WEIGHTS_NAME
    if direct.is_file():
        return direct.parent

    ckpt_root = run_dir / CHECKPOINTS_DIR_NAME
    if not ckpt_root.is_dir():
        raise FileNotFoundError(
            f"No checkpoints/ under run_dir and no direct pretrained_model/: {run_dir}"
        )

    spec = checkpoint.strip()
    if spec in ("last", CHECKPOINT_LAST_ALIAS):
        latest = _read_summary_latest(run_dir)
        if latest:
            return _resolve_checkpoint_alias(run_dir, Path(latest).name)
        return _resolve_checkpoint_alias(run_dir, CHECKPOINT_LAST_ALIAS)

    if spec.startswith("step:"):
        step_name = spec.split(":", 1)[1].strip()
        return _resolve_step_dir(ckpt_root, step_name)

    if spec.endswith(".pt"):
        return _resolve_checkpoint_alias(run_dir, spec)

    return _resolve_step_dir(ckpt_root, spec)


def is_run_directory(path: Path) -> bool:
    """True when path looks like a lerobot_trainer run root."""
    if (path / PRETRAINED_MODEL_DIR / MODEL_WEIGHTS_NAME).is_file():
        return True
    if (path / CHECKPOINTS_DIR_NAME).is_dir():
        return True
    return any((path / name).is_file() for name in _RUN_ARTIFACT_NAMES)
