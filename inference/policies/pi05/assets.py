"""Immutable local asset validation for offline PI0.5 inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PI05_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
_TOKENIZER_MARKERS = ("tokenizer_config.json", "tokenizer.json", "spiece.model")


def resolve_required_local_dir(raw: str | None, param_name: str) -> Path:
    if raw is None or str(raw).strip() in ("", "null", "None"):
        raise ValueError(f"{param_name} is required (pass an existing local directory path).")
    path = Path(str(raw).strip()).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{param_name} not found or not a directory: {path}")
    return path


def is_tokenizer_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in _TOKENIZER_MARKERS)


def validate_tokenizer_reference(name: str) -> str:
    resolved = Path(name).expanduser().resolve()
    if not is_tokenizer_dir(resolved):
        raise FileNotFoundError(
            f"Tokenizer directory invalid: {resolved}. "
            "Expected tokenizer_config.json, tokenizer.json, or spiece.model."
        )
    return str(resolved)


def _reject_obsolete_processor_schema(pretrained_dir: Path) -> None:
    preprocessor_path = pretrained_dir / "policy_preprocessor.json"
    data = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    obsolete = [
        step.get("registry_name")
        for step in data.get("steps", [])
        if step.get("registry_name") == "delta_actions_processor"
    ]
    if obsolete:
        raise ValueError(
            f"Obsolete PI0.5 processor schema in {preprocessor_path}: "
            "delta_actions_processor must be migrated to a LeRobot 0.6-compatible "
            "processor artifact before inference. Runtime loading never modifies checkpoints."
        )


def ensure_pi05_offline_assets(
    pretrained_path: Path | str | None,
    tokenizer_path: str | None = None,
) -> tuple[Path, str]:
    """Validate immutable local PI0.5 policy and tokenizer artifacts."""
    path = resolve_required_local_dir(
        str(pretrained_path) if pretrained_path is not None else None,
        "policy.pretrained_path",
    )
    missing = [name for name in _PI05_REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"PI0.5 pretrained directory {path} is missing files: {missing}")

    local_tokenizer = validate_tokenizer_reference(
        str(resolve_required_local_dir(tokenizer_path, "policy.tokenizer_path"))
    )
    _reject_obsolete_processor_schema(path)
    logger.info("PI0.5 offline assets OK: weights=%s tokenizer=%s", path, local_tokenizer)
    return path, local_tokenizer


def load_local_tokenizer(tokenizer_path: str) -> Any:
    """Load one tokenizer instance without network access or process-global patches."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
