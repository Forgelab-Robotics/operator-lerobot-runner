#!/usr/bin/env python3
"""Run one inference step locally (no Dora) for smoke testing."""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from lerobot_inference.inference.config import load_config
from lerobot_inference.inference.policies.registry import create_policy_adapter


def _build_observation(
    config,
    args,
    camera_aliases: list[str] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    state_dim = args.state_dim or len(config.state_joint_order)
    observation: dict[str, np.ndarray] = {
        "observation.state": rng.normal(size=state_dim).astype(np.float32),
    }
    aliases = config.alias_for_cameras if camera_aliases is None else camera_aliases
    for alias in aliases:
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
    image_inputs = config.image_inputs_for(policy.required_image_keys)
    camera_aliases = list(image_inputs.values())
    observation = _build_observation(config, args, camera_aliases)
    timeout = float(getattr(args, "async_timeout", 120.0))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("--async-timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    try:
        action = None
        while action is None:
            action = policy.generate_action(observation, camera_aliases)
            if action is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"async policy did not produce an action within {timeout:.1f}s"
                )
            time.sleep(0.05)
        payload = {
            "policy_type": config.policy.get("type"),
            "pretrained_path": config.runtime_policy_config().get("pretrained_path"),
            "action": action.tolist(),
            "action_dim": int(action.shape[0]),
        }
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        stop = getattr(policy, "stop", None)
        if callable(stop):
            stop()


def main() -> int:
    """兼容旧入口：转发到统一 CLI 的 infer-once 子命令。"""
    from lerobot_inference.cli import main_infer_once

    return main_infer_once()


if __name__ == "__main__":
    sys.exit(main())
