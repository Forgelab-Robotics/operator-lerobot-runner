"""Offline-only Pi0.5 tokenizer / processor helpers (self-contained)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PI05_WEIGHTS_FILE = "model.safetensors"
_TOKENIZER_MARKERS = ("tokenizer_config.json", "tokenizer.json", "spiece.model")
_PI0_POLICY_TYPES = frozenset({"pi05", "pi0", "pi0_fast"})

_tokenizer_path_override: str | None = None
_original_auto_tokenizer_from_pretrained = None


def is_hub_model_id(name: str) -> bool:
    path = Path(name).expanduser()
    if path.exists():
        return False
    return "/" in name and not name.startswith(("/", "./", "../", "~"))


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
    if is_hub_model_id(name):
        raise ValueError(
            f"Tokenizer still points at HuggingFace hub id {name!r}. "
            "Set policy.tokenizer_path to an existing local directory."
        )
    resolved = Path(name).expanduser().resolve()
    if not is_tokenizer_dir(resolved):
        raise FileNotFoundError(
            f"Tokenizer directory invalid: {resolved}. "
            "Expected tokenizer_config.json, tokenizer.json, or spiece.model."
        )
    return str(resolved)


def _resolve_tokenizer_for_offline(name: str) -> str:
    if _tokenizer_path_override is None:
        raise ValueError(
            "policy.tokenizer_path is required for Pi0.x offline inference. "
            "Pass an existing local tokenizer directory; hub ids are not allowed."
        )
    local = validate_tokenizer_reference(
        str(resolve_required_local_dir(_tokenizer_path_override, "policy.tokenizer_path"))
    )
    if is_hub_model_id(name) or str(Path(name).expanduser().resolve()) != Path(local).resolve():
        logger.warning(
            "Offline mode: replacing tokenizer reference %r with policy.tokenizer_path %s",
            name,
            local,
        )
    return local


def set_tokenizer_path(raw: str | None) -> None:
    global _tokenizer_path_override
    if raw is None or str(raw).strip() in ("", "null", "None"):
        _tokenizer_path_override = None
    else:
        _tokenizer_path_override = str(raw).strip()


def require_local_tokenizer_path() -> str:
    if _tokenizer_path_override is None:
        raise ValueError(
            "policy.tokenizer_path is required "
            "(existing local directory with tokenizer files)."
        )
    return validate_tokenizer_reference(
        str(resolve_required_local_dir(_tokenizer_path_override, "policy.tokenizer_path"))
    )


def patch_pi05_policy_files(pretrained_dir: Path) -> None:
    import json

    preprocessor = pretrained_dir / "policy_preprocessor.json"
    if not preprocessor.is_file():
        return

    data = json.loads(preprocessor.read_text(encoding="utf-8"))
    changed = False
    local_tok = require_local_tokenizer_path()

    for step in data.get("steps", []):
        registry_name = step.get("registry_name")
        if registry_name == "delta_actions_processor":
            step["registry_name"] = "relative_actions_processor"
            changed = True
        if step.get("registry_name") == "tokenizer_processor":
            step.setdefault("config", {})["tokenizer_name"] = local_tok
            changed = True

    if changed:
        preprocessor.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        logger.info("Patched policy_preprocessor.json in %s", pretrained_dir)


def _merge_tokenizer_override(overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(overrides or {})
    local = require_local_tokenizer_path()
    tok_cfg = dict(merged.get("tokenizer_processor") or {})
    tok_cfg["tokenizer_name"] = local
    merged["tokenizer_processor"] = tok_cfg
    logger.info("Using local Paligemma tokenizer: %s", local)
    return merged


def patch_auto_tokenizer_offline() -> None:
    global _original_auto_tokenizer_from_pretrained
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    if getattr(AutoTokenizer, "_lerobot_inference_offline_patched", False):
        return

    if _original_auto_tokenizer_from_pretrained is None:
        _original_auto_tokenizer_from_pretrained = AutoTokenizer.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("local_files_only", True)
        resolved = _resolve_tokenizer_for_offline(str(pretrained_model_name_or_path))
        return _original_auto_tokenizer_from_pretrained(cls, resolved, *args, **kwargs)

    AutoTokenizer.from_pretrained = from_pretrained
    AutoTokenizer._lerobot_inference_offline_patched = True


def patch_make_pre_post_processors() -> None:
    from lerobot.policies import factory as policy_factory

    if getattr(policy_factory, "_lerobot_inference_patched", False):
        return

    original = policy_factory.make_pre_post_processors

    def wrapped(policy_cfg, pretrained_path=None, **kwargs):
        policy_type = getattr(policy_cfg, "type", None)
        if pretrained_path and policy_type in _PI0_POLICY_TYPES:
            merged = _merge_tokenizer_override(kwargs.get("preprocessor_overrides"))
            kwargs["preprocessor_overrides"] = merged
        return original(policy_cfg, pretrained_path=pretrained_path, **kwargs)

    policy_factory.make_pre_post_processors = wrapped
    policy_factory._lerobot_inference_patched = True


def ensure_pi05_offline_assets(
    pretrained_path: Path | str | None,
    tokenizer_path: str | None = None,
) -> None:
    set_tokenizer_path(tokenizer_path)
    path = resolve_required_local_dir(
        str(pretrained_path) if pretrained_path is not None else None,
        "policy.pretrained_path",
    )

    weights = path / _PI05_WEIGHTS_FILE
    if not weights.is_file():
        raise FileNotFoundError(
            f"Pretrained weights not found: {weights}. "
            f"Expected {_PI05_WEIGHTS_FILE} under policy.pretrained_path."
        )

    patch_pi05_policy_files(path)
    patch_auto_tokenizer_offline()
    patch_make_pre_post_processors()
    local_tok = require_local_tokenizer_path()
    logger.info("Pi0.x offline assets OK: weights=%s tokenizer=%s", path, local_tok)
