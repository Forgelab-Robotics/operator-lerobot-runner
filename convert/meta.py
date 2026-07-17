"""Load conversion metadata from policy_train artifacts."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 对齐 pick_and_place / single_real_inference / examples/dora_convert/minimal_task.json
_DEFAULT_CAMERA_NAMES: tuple[str, ...] = (
    "observation.images.left",
    "observation.images.right",
    "observation.images.top",
)


@dataclass(frozen=True)
class PolicyTrainMeta:
    src_dir: Path
    checkpoint_path: Path
    stats_path: Path
    state_dim: int
    action_dim: int
    camera_names: list[str]
    policy_params: dict[str, Any]
    task_id: str | None = None


def _load_pickle_stats(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        stats = pickle.load(handle)
    required = ("qpos_mean", "qpos_std", "action_mean", "action_std")
    missing = [key for key in required if key not in stats]
    if missing:
        raise KeyError(f"dataset_stats.pkl missing keys: {missing}")
    return stats


def _infer_dims(stats: dict[str, np.ndarray]) -> tuple[int, int]:
    state_dim = int(np.asarray(stats["qpos_mean"]).reshape(-1).shape[0])
    action_dim = int(np.asarray(stats["action_mean"]).reshape(-1).shape[0])
    return state_dim, action_dim


def _load_task_json(task_json: Path | None) -> dict[str, Any]:
    if task_json is None or not task_json.is_file():
        return {}
    return json.loads(task_json.read_text(encoding="utf-8"))


def _extract_policy_params(task_data: dict[str, Any]) -> dict[str, Any]:
    launch = task_data.get("launch") or {}
    layered = launch.get("layered_request") or {}
    policy = layered.get("policy") or {}
    params = dict(policy.get("params") or {})
    if launch.get("cameras"):
        params.setdefault("camera_names", launch["cameras"])
    if launch.get("cameras") and "cameras" not in params:
        params["cameras"] = launch["cameras"]
    return params


def _default_policy_params() -> dict[str, Any]:
    """Defaults aligned with pick_and_place ACT + policy_train common task.json."""
    return {
        "lr": 2e-5,
        "lr_backbone": 2e-5,
        "chunk_size": 100,
        "num_queries": 100,
        "query_size": 50,
        "kl_weight": 10,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": list(_DEFAULT_CAMERA_NAMES),
        "cameras": list(_DEFAULT_CAMERA_NAMES),
    }


def _normalize_camera_name(name: str) -> str:
    """Bare alias ``left`` → ``observation.images.left``；已是完整 key 则原样返回。"""
    text = str(name).strip()
    if not text:
        raise ValueError("camera name must be non-empty")
    if text.startswith("observation.images."):
        return text
    return f"observation.images.{text}"


def _resolve_camera_names(policy_params: dict[str, Any]) -> list[str]:
    raw = list(
        policy_params.get("camera_names")
        or policy_params.get("cameras")
        or []
    )
    if not raw:
        logger.warning(
            "task_json 未提供相机名，使用默认三路相机: %s",
            ", ".join(_DEFAULT_CAMERA_NAMES),
        )
        raw = list(_DEFAULT_CAMERA_NAMES)
    return [_normalize_camera_name(name) for name in raw]


def load_policy_train_meta(
    src_dir: str | Path,
    *,
    checkpoint_path: Path,
    task_json: str | Path | None = None,
) -> tuple[PolicyTrainMeta, dict[str, np.ndarray]]:
    src_dir = Path(src_dir).expanduser().resolve()
    stats_path = src_dir / "dataset_stats.pkl"
    if not stats_path.is_file():
        raise FileNotFoundError(f"dataset_stats.pkl not found under {src_dir}")

    stats = _load_pickle_stats(stats_path)
    state_dim, action_dim = _infer_dims(stats)

    task_path = Path(task_json).expanduser().resolve() if task_json else None
    task_data = _load_task_json(task_path)
    policy_params = _default_policy_params()
    policy_params.update(_extract_policy_params(task_data))

    camera_names = _resolve_camera_names(policy_params)
    policy_params["camera_names"] = camera_names
    policy_params["cameras"] = camera_names

    task_id = task_data.get("task_id")
    return (
        PolicyTrainMeta(
            src_dir=src_dir,
            checkpoint_path=checkpoint_path,
            stats_path=stats_path,
            state_dim=state_dim,
            action_dim=action_dim,
            camera_names=camera_names,
            policy_params=policy_params,
            task_id=task_id,
        ),
        stats,
    )
