#!/usr/bin/env python3
"""Deterministic Forge protocol simulator and observation probe for FastWAM examples."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from forge_msgs import Image, JointCommand, JointState, PolicyCommand, PolicyCommandStatus

logger = logging.getLogger("fastwam_protocol")

_CONTROL_COMMANDS = ("start", "pause", "resume", "reset", "stop")


@dataclass(frozen=True)
class ImageSpec:
    input_id: str
    width: int
    height: int
    pattern: int


@dataclass(frozen=True)
class ProtocolConfig:
    path: Path
    policy_id: str
    control_dir: Path
    state_joints: tuple[str, ...]
    action_joints: tuple[str, ...]
    initial_state: tuple[float, ...]
    action_to_state: dict[str, str]
    images: tuple[ImageSpec, ...]
    image_every_ticks: int
    state_clip: tuple[float, float]


def _non_empty_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a non-empty list of strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(value)


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    config_path = Path(path).expanduser().resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Protocol config must be a mapping: {config_path}")

    state_joints = _non_empty_strings(data.get("state_joints"), "state_joints")
    action_joints = _non_empty_strings(data.get("action_joints"), "action_joints")
    initial_raw = data.get("initial_state", [0.0] * len(state_joints))
    if not isinstance(initial_raw, list) or len(initial_raw) != len(state_joints):
        raise ValueError("initial_state length must equal state_joints length")
    initial_state = tuple(float(value) for value in initial_raw)
    if not np.isfinite(initial_state).all():
        raise ValueError("initial_state must contain only finite values")

    mapping = data.get("action_to_state")
    if not isinstance(mapping, dict) or set(mapping) != set(action_joints):
        raise ValueError("action_to_state must map every action joint exactly once")
    action_to_state = {str(action): str(state) for action, state in mapping.items()}
    unknown_states = set(action_to_state.values()) - set(state_joints)
    if unknown_states:
        raise ValueError(f"action_to_state references unknown states: {sorted(unknown_states)}")

    images_raw = data.get("images")
    if not isinstance(images_raw, dict) or not images_raw:
        raise ValueError("images must be a non-empty mapping")
    images: list[ImageSpec] = []
    for index, (input_id, raw) in enumerate(images_raw.items()):
        if not isinstance(raw, dict):
            raise ValueError(f"images.{input_id} must be a mapping")
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"images.{input_id} width/height must be positive")
        images.append(
            ImageSpec(
                input_id=str(input_id),
                width=width,
                height=height,
                pattern=int(raw.get("pattern", index)),
            )
        )

    image_every_ticks = int(data.get("image_every_ticks", 5))
    if image_every_ticks <= 0:
        raise ValueError("image_every_ticks must be positive")
    clip_raw = data.get("state_clip", [-10.0, 10.0])
    if not isinstance(clip_raw, list) or len(clip_raw) != 2:
        raise ValueError("state_clip must contain [minimum, maximum]")
    state_clip = (float(clip_raw[0]), float(clip_raw[1]))
    if not np.isfinite(state_clip).all() or state_clip[0] >= state_clip[1]:
        raise ValueError("state_clip must contain finite increasing bounds")

    control_value = Path(str(data.get("control_dir", "control"))).expanduser()
    control_dir = (
        control_value.resolve()
        if control_value.is_absolute()
        else (config_path.parent / control_value).resolve()
    )
    policy_id = str(data.get("policy_id", "")).strip()
    if not policy_id:
        raise ValueError("policy_id must be non-empty")
    return ProtocolConfig(
        path=config_path,
        policy_id=policy_id,
        control_dir=control_dir,
        state_joints=state_joints,
        action_joints=action_joints,
        initial_state=initial_state,
        action_to_state=action_to_state,
        images=tuple(images),
        image_every_ticks=image_every_ticks,
        state_clip=state_clip,
    )


def make_deterministic_image(spec: ImageSpec) -> np.ndarray:
    """Create a stable RGB pattern that makes camera swaps visible."""
    x = np.arange(spec.width, dtype=np.uint16)[None, :]
    y = np.arange(spec.height, dtype=np.uint16)[:, None]
    frame = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    frame[..., 0] = (x + spec.pattern * 53) % 256
    frame[..., 1] = (y + spec.pattern * 97) % 256
    frame[..., 2] = ((x // 2 + y // 2) + spec.pattern * 29) % 256
    return frame


def apply_joint_command(
    config: ProtocolConfig,
    state: np.ndarray,
    command: JointCommand,
) -> np.ndarray:
    if command.name != list(config.action_joints):
        raise ValueError(
            "JointCommand names/order do not match simulator action_joints: "
            f"received={command.name}, expected={list(config.action_joints)}"
        )
    if len(command.position) != len(config.action_joints):
        raise ValueError("JointCommand must provide one position value per action joint")
    values = command.to_np(list(config.action_joints), field="position")
    if values.shape != (len(config.action_joints),) or not np.isfinite(values).all():
        raise ValueError("JointCommand position must have the configured finite action shape")
    updated = np.array(state, dtype=np.float64, copy=True)
    state_index = {name: index for index, name in enumerate(config.state_joints)}
    for action_name, value in zip(config.action_joints, values, strict=True):
        updated[state_index[config.action_to_state[action_name]]] = np.clip(
            value,
            config.state_clip[0],
            config.state_clip[1],
        )
    return updated


def pending_control_commands(config: ProtocolConfig) -> list[tuple[str, Path]]:
    config.control_dir.mkdir(parents=True, exist_ok=True)
    return [
        (command, config.control_dir / command)
        for command in _CONTROL_COMMANDS
        if (config.control_dir / command).is_file()
    ]


def make_policy_command(config: ProtocolConfig, command: str) -> PolicyCommand:
    if command not in _CONTROL_COMMANDS:
        raise ValueError(f"Unsupported example control command: {command}")
    return PolicyCommand.from_inputs(
        policy_id=config.policy_id,
        command=command,
        request_id=f"{command}-{time.time_ns()}",
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def persist_command_status(config: ProtocolConfig, status: PolicyCommandStatus) -> None:
    write_json_atomic(
        config.control_dir / "status.json",
        {
            "policy_id": status.policy_id,
            "command": status.command,
            "request_id": status.request_id,
            "status": status.status,
            "message": status.message,
            "outputs": status.outputs(),
        },
    )


def persist_action(config: ProtocolConfig, command: JointCommand, count: int) -> None:
    write_json_atomic(
        config.control_dir / "last_action.json",
        {
            "count": count,
            "name": command.name,
            "position": command.position,
            "finite": bool(np.isfinite(command.position).all()),
        },
    )


def run_simulator(config: ProtocolConfig) -> int:
    from dora import Node

    node = Node()
    state = np.array(config.initial_state, dtype=np.float64)
    frames = {
        spec.input_id: Image.from_numpy(make_deterministic_image(spec), encoding="rgb8")
        for spec in config.images
    }
    tick_count = 0
    action_count = 0
    logger.info(
        "protocol simulator ready: state=%d action=%d images=%s",
        len(config.state_joints),
        len(config.action_joints),
        [spec.input_id for spec in config.images],
    )
    for event in node:
        event_type = event.get("type")
        if event_type == "INPUT":
            input_id = event["id"]
            value = event.get("value")
            if input_id == "action" and value is not None:
                command = JointCommand.from_arrow(value)
                state = apply_joint_command(config, state, command)
                action_count += 1
                persist_action(config, command, action_count)
                if action_count == 1:
                    logger.info("complete JointCommand loop received")
                continue
            if input_id == "policy_command_status" and value is not None:
                status = PolicyCommandStatus.from_arrow(value)
                persist_command_status(config, status)
                logger.info("policy command %s: %s", status.command, status.status)
                continue
            if input_id != "tick":
                continue

            tick_count += 1
            node.send_output(
                "proprio_state",
                JointState.from_np(state, list(config.state_joints)).to_arrow(),
            )
            if tick_count == 1 or tick_count % config.image_every_ticks == 0:
                for output_id, image in frames.items():
                    node.send_output(output_id, image.to_arrow())
            for command_name, trigger in pending_control_commands(config):
                node.send_output(
                    "policy_command",
                    make_policy_command(config, command_name).to_arrow(),
                )
                trigger.unlink(missing_ok=True)
                logger.info("sent policy command: %s", command_name)
        elif event_type in {"STOP", "ERROR"}:
            break
    return 0


def run_probe(config: ProtocolConfig) -> int:
    from dora import Node

    node = Node()
    state_ready = False
    images_ready: set[str] = set()
    expected_images = {spec.input_id: spec for spec in config.images}
    announced = False
    for event in node:
        event_type = event.get("type")
        if event_type == "INPUT":
            input_id = event["id"]
            value = event.get("value")
            if value is None:
                continue
            if input_id == "proprio_state":
                state = JointState.from_arrow(value)
                vector = state.to_np(list(config.state_joints))
                if vector.shape != (len(config.state_joints),) or not np.isfinite(vector).all():
                    raise ValueError("probe received an invalid proprio_state")
                state_ready = True
            elif input_id in expected_images:
                spec = expected_images[input_id]
                image = Image.from_arrow(value)
                if (
                    image.encoding != "rgb8"
                    or image.width != spec.width
                    or image.height != spec.height
                ):
                    raise ValueError(
                        f"probe image mismatch for {input_id}: "
                        f"{image.encoding} {image.width}x{image.height}"
                    )
                images_ready.add(input_id)
            if state_ready and images_ready == set(expected_images) and not announced:
                logger.info(
                    "OBSERVATION_READY state_dim=%d images=%s",
                    len(config.state_joints),
                    sorted(images_ready),
                )
                announced = True
        elif event_type in {"STOP", "ERROR"}:
            break
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("simulator", "probe"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_protocol_config(args.config)
    return run_simulator(config) if args.mode == "simulator" else run_probe(config)


if __name__ == "__main__":
    raise SystemExit(main())
