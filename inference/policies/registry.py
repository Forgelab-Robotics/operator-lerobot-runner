"""Extensible registry for LeRobot policy inference adapters."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from lerobot_inference.inference.artifact_resolver import resolve_pretrained_path
from lerobot_inference.inference.policies.act import ACTPolicyAdapter
from lerobot_inference.inference.policies.base import LerobotPolicyAdapter
from lerobot_inference.inference.policies.pi05 import PI05PolicyAdapter

PolicyFactory = Callable[[dict[str, Any], str], LerobotPolicyAdapter]

_POLICY_ALIASES: dict[str, str] = {
    "act": "act",
    "ACT": "act",
    "pi05": "pi05",
    "pi0.5": "pi05",
    "PI05": "pi05",
    "smolvla": "smolvla",
    "SmolVLA": "smolvla",
    "pi0": "pi0",
    "pi0_fast": "pi0_fast",
    "diffusion": "diffusion",
    "vqbet": "vqbet",
    "tdmpc": "tdmpc",
}

# Implemented adapters keyed by normalized LeRobot policy type.
_IMPLEMENTED: dict[str, PolicyFactory] = {}

# Types LeRobot supports but this package has not wired yet.
_PLANNED_LEROBOT_TYPES = frozenset(
    {
        "pi0",
        "pi0_fast",
        "smolvla",
        "diffusion",
        "vqbet",
        "tdmpc",
        "xvla",
        "groot",
    }
)


def normalize_policy_type(raw: str) -> str:
    key = raw.strip()
    if key not in _POLICY_ALIASES:
        raise ValueError(
            f"Unsupported policy.type: {raw!r}. "
            f"Known aliases: {sorted(_POLICY_ALIASES)}"
        )
    return _POLICY_ALIASES[key]


def register_policy_type(policy_type: str, factory: PolicyFactory) -> None:
    normalized = normalize_policy_type(policy_type)
    _IMPLEMENTED[normalized] = factory


def supported_policy_types() -> list[str]:
    return sorted(_IMPLEMENTED)


def planned_policy_types() -> list[str]:
    return sorted(_PLANNED_LEROBOT_TYPES - set(_IMPLEMENTED))


def _expected_image_keys(policy_config: dict[str, Any], camera_aliases: list[str]) -> set[str]:
    explicit = policy_config.get("expected_image_keys")
    if explicit:
        return {str(key) for key in explicit}
    return {f"observation.images.{alias}" for alias in camera_aliases}


def _act_runtime_dimensions(
    policy_config: dict[str, Any],
) -> tuple[int | None, int | None]:
    state_dim = policy_config.get("state_joint_count")
    if state_dim is None:
        state_dim = policy_config.get("state_dim", policy_config.get("joint_count"))
    action_dim = policy_config.get("action_joint_count")
    if action_dim is None:
        action_dim = policy_config.get("action_dim", policy_config.get("joint_count"))
    # Historical runtime dictionaries used either state_dim or joint_count as
    # the shared ACT state/action dimension.
    if state_dim is None:
        state_dim = action_dim
    if action_dim is None:
        action_dim = state_dim
    return (
        int(state_dim) if state_dim is not None else None,
        int(action_dim) if action_dim is not None else None,
    )


def _act_policy_config_overrides(policy_config: dict[str, Any]) -> dict[str, Any]:
    if "temporal_ensemble_coeff" not in policy_config:
        return {}
    raw_coeff = policy_config["temporal_ensemble_coeff"]
    if raw_coeff is None:
        return {"temporal_ensemble_coeff": None}
    if isinstance(raw_coeff, bool) or not isinstance(raw_coeff, (int, float)):
        raise ValueError("policy.temporal_ensemble_coeff must be a finite number or null")
    coeff = float(raw_coeff)
    if not math.isfinite(coeff):
        raise ValueError("policy.temporal_ensemble_coeff must be a finite number or null")
    # LeRobot requires one fresh prediction per step when temporal ensembling is enabled.
    return {"temporal_ensemble_coeff": coeff, "n_action_steps": 1}


def _create_act(policy_config: dict[str, Any], pretrained_path: str) -> LerobotPolicyAdapter:
    from lerobot_inference.inference.compile_utils import compile_enabled_from_policy_config

    camera_aliases = list(policy_config.get("camera_names") or [])
    adapter = ACTPolicyAdapter.from_pretrained(
        pretrained_path,
        device=policy_config.get("device"),
        expected_image_keys=_expected_image_keys(policy_config, camera_aliases),
        torch_compile=compile_enabled_from_policy_config(policy_config),
        policy_config_overrides=_act_policy_config_overrides(policy_config),
    )
    state_dim, action_dim = _act_runtime_dimensions(policy_config)
    if state_dim is not None and action_dim is not None:
        adapter.validate_io_dimensions(state_dim, action_dim)
    return adapter


register_policy_type("act", _create_act)


def _create_pi05(policy_config: dict[str, Any], pretrained_path: str) -> LerobotPolicyAdapter:
    tokenizer_path = policy_config.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("policy.tokenizer_path is required for Pi0.5 inference.")
    camera_aliases = list(policy_config.get("camera_names") or [])
    return PI05PolicyAdapter.from_pretrained(
        pretrained_path,
        tokenizer_path=str(tokenizer_path),
        device=policy_config.get("device"),
        instruction=str(policy_config.get("instruction", "")),
        expected_image_keys=_expected_image_keys(policy_config, camera_aliases),
        get_actions_threshold=int(policy_config.get("get_actions_threshold", 0)),
    )


register_policy_type("pi05", _create_pi05)


def resolve_policy_pretrained_path(policy_config: dict[str, Any]) -> str:
    # LeRobot 目录（推荐）
    if policy_config.get("pretrained_path"):
        return str(resolve_pretrained_path(pretrained_path=policy_config["pretrained_path"]))
    # 兼容别名：部分用户沿用 pick_and_place 字段名，但值必须是 pretrained_model 目录
    if policy_config.get("ckpt_path"):
        return str(resolve_pretrained_path(pretrained_path=policy_config["ckpt_path"]))
    if policy_config.get("run_dir"):
        return str(
            resolve_pretrained_path(
                run_dir=policy_config["run_dir"],
                checkpoint=str(policy_config.get("checkpoint", "last")),
            )
        )
    raise ValueError(
        "policy.pretrained_path（或 ckpt_path / run_dir）必填，"
        "指向含 model.safetensors 的 pretrained_model 目录。"
    )


def create_policy_adapter(policy_config: dict[str, Any]) -> LerobotPolicyAdapter:
    """Instantiate a registered policy adapter from runtime policy config."""
    policy_type = normalize_policy_type(str(policy_config["type"]))
    factory = _IMPLEMENTED.get(policy_type)
    if factory is None:
        if policy_type in _PLANNED_LEROBOT_TYPES:
            raise NotImplementedError(
                f"Policy type {policy_type!r} is supported by LeRobot but not yet implemented "
                f"in lerobot_inference. Implemented: {supported_policy_types()}. "
                f"Add an adapter under lerobot_inference/inference/policies/ and register it."
            )
        raise ValueError(
            f"Unknown policy type {policy_type!r}. Implemented: {supported_policy_types()}"
        )
    pretrained_path = resolve_policy_pretrained_path(policy_config)
    policy_config = {**policy_config, "pretrained_path": pretrained_path}
    return factory(policy_config, pretrained_path)
