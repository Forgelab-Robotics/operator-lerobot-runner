#!/usr/bin/env python3
"""Run one inference step locally (no Dora) for smoke testing."""

from __future__ import annotations

import json
import sys

import numpy as np

from lerobot_inference.inference.config import load_config
from lerobot_inference.inference.policies.registry import create_policy_adapter


def _build_observation(config, args) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    state_dim = args.state_dim or len(config.joint_order)
    observation: dict[str, np.ndarray] = {
        "observation.state": rng.normal(size=state_dim).astype(np.float32),
    }
    for alias in config.alias_for_cameras:
        observation[f"observation.images.{alias}"] = rng.integers(
            0,
            255,
            size=(args.height, args.width, 3),
            dtype=np.uint8,
        )
    return observation


def run_infer_once(args) -> int:
    """执行单步 smoke；args 需含 config，以及可选 state_dim/height/width/seed。"""
    config = load_config(config_path=args.config)
    policy = create_policy_adapter(config.runtime_policy_config())
    policy.reset()
    observation = _build_observation(config, args)
    action = policy.generate_action(observation, config.alias_for_cameras)
    payload = {
        "policy_type": config.policy.get("type"),
        "pretrained_path": config.runtime_policy_config().get("pretrained_path"),
        "action": action.tolist(),
        "action_dim": int(action.shape[0]),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    """兼容旧入口：转发到统一 CLI 的 infer-once 子命令。"""
    from lerobot_inference.cli import main_infer_once

    return main_infer_once()


if __name__ == "__main__":
    sys.exit(main())
