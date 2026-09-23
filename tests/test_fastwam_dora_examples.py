from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from forge_msgs import JointCommand, PolicyCommandStatus

from lerobot_inference.inference.config import PolicyNodeConfig

_ROOT = Path(__file__).resolve().parents[1]
_SUPPORT = _ROOT / "examples/fastwam_dora_support"


def _load_support_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SUPPORT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol_node = _load_support_module("fastwam_example_protocol_node", "protocol_node.py")

apply_joint_command = protocol_node.apply_joint_command
load_protocol_config = protocol_node.load_protocol_config
make_deterministic_image = protocol_node.make_deterministic_image
make_policy_command = protocol_node.make_policy_command
pending_control_commands = protocol_node.pending_control_commands
persist_command_status = protocol_node.persist_command_status

_EXAMPLES = {
    "libero": _ROOT / "examples/dora_sim_infer_fastwam_libero",
    "robotwin": _ROOT / "examples/dora_sim_infer_fastwam_robotwin",
}


@pytest.mark.parametrize(
    ("name", "state_dim", "action_dim", "images"),
    [
        (
            "libero",
            8,
            7,
            {"image/camera_0": (224, 224), "image/camera_1": (224, 224)},
        ),
        ("robotwin", 14, 14, {"image/combined": (320, 384)}),
    ],
)
def test_fastwam_protocol_configs(
    name: str,
    state_dim: int,
    action_dim: int,
    images: dict[str, tuple[int, int]],
) -> None:
    config = load_protocol_config(_EXAMPLES[name] / "simulator.yaml")

    assert len(config.state_joints) == state_dim
    assert len(config.action_joints) == action_dim
    assert {
        spec.input_id: (spec.width, spec.height) for spec in config.images
    } == images
    for spec in config.images:
        frame = make_deterministic_image(spec)
        assert frame.shape == (spec.height, spec.width, 3)
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, make_deterministic_image(spec))


def test_libero_action_maps_7d_into_8d_state() -> None:
    config = load_protocol_config(_EXAMPLES["libero"] / "simulator.yaml")
    state = np.full(8, 9.0)
    command = JointCommand(
        name=list(config.action_joints),
        position=[float(index) for index in range(7)],
    )

    updated = apply_joint_command(config, state, command)

    np.testing.assert_array_equal(updated[:7], np.arange(7, dtype=np.float64))
    assert updated[7] == 9.0


def test_robotwin_action_maps_14d_one_to_one_and_clips() -> None:
    config = load_protocol_config(_EXAMPLES["robotwin"] / "simulator.yaml")
    command = JointCommand(
        name=list(config.action_joints),
        position=[20.0] * 14,
    )

    updated = apply_joint_command(config, np.zeros(14), command)

    np.testing.assert_array_equal(updated, np.full(14, 10.0))


def test_protocol_rejects_wrong_or_non_finite_action() -> None:
    config = load_protocol_config(_EXAMPLES["libero"] / "simulator.yaml")
    with pytest.raises(ValueError, match="names/order"):
        apply_joint_command(
            config,
            np.zeros(8),
            JointCommand(name=list(reversed(config.action_joints)), position=[0.0] * 7),
        )
    with pytest.raises(ValueError, match="one position value"):
        apply_joint_command(
            config,
            np.zeros(8),
            JointCommand(name=list(config.action_joints)),
        )
    values = [0.0] * 7
    values[-1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        apply_joint_command(
            config,
            np.zeros(8),
            JointCommand(name=list(config.action_joints), position=values),
        )


def test_manual_control_trigger_and_status_file(tmp_path: Path) -> None:
    source = yaml.safe_load((_EXAMPLES["libero"] / "simulator.yaml").read_text())
    source["control_dir"] = str(tmp_path / "control")
    config_path = tmp_path / "simulator.yaml"
    config_path.write_text(yaml.safe_dump(source))
    config = load_protocol_config(config_path)
    trigger = config.control_dir / "start"
    trigger.parent.mkdir(parents=True)
    trigger.touch()

    assert pending_control_commands(config) == [("start", trigger)]
    command = make_policy_command(config, "start")
    assert command.policy_id == config.policy_id
    assert command.command == "start"

    status = PolicyCommandStatus.from_outputs(
        policy_id=config.policy_id,
        command="start",
        request_id=command.request_id,
        status="done",
        outputs={"phase": "running"},
    )
    persist_command_status(config, status)
    payload = json.loads((config.control_dir / "status.json").read_text())
    assert payload["status"] == "done"
    assert payload["outputs"] == {"phase": "running"}


@pytest.mark.parametrize("name", ["libero", "robotwin"])
def test_manual_and_auto_policy_configs(name: str) -> None:
    directory = _EXAMPLES[name]
    manual = PolicyNodeConfig.from_yaml_path(directory / "policy_fastwam.yaml")
    automatic = PolicyNodeConfig.from_yaml_path(directory / "policy_fastwam_auto.yaml")
    protocol = load_protocol_config(directory / "simulator.yaml")

    assert manual.auto_start is False
    assert automatic.auto_start is True
    assert manual.policy["policy_id"] == protocol.policy_id
    assert automatic.policy["policy_id"] == protocol.policy_id
    assert manual.state_joint_order == list(protocol.state_joints)
    assert manual.joint_order == list(protocol.action_joints)
    assert set(manual.image_input_id_to_alias) == {
        spec.input_id for spec in protocol.images
    }


def _node(dataflow: dict, node_id: str) -> dict:
    return next(node for node in dataflow["nodes"] if node["id"] == node_id)


@pytest.mark.parametrize("name", ["libero", "robotwin"])
def test_fastwam_dataflow_topologies(name: str) -> None:
    directory = _EXAMPLES[name]
    protocol = load_protocol_config(directory / "simulator.yaml")
    image_ids = {spec.input_id for spec in protocol.images}

    simulator_flow = yaml.safe_load((directory / "dataflow_simulator.yaml").read_text())
    simulator_only = _node(simulator_flow, "simulator")
    probe = _node(simulator_flow, "observation_probe")
    assert simulator_only["path"] == "../fastwam_dora_support/project_python.sh"
    assert probe["path"] == "../fastwam_dora_support/project_python.sh"
    assert set(probe["inputs"]) == {"proprio_state", *image_ids}

    for filename, policy_config in (
        ("dataflow.yaml", "./policy_fastwam.yaml"),
        ("dataflow_auto.yaml", "./policy_fastwam_auto.yaml"),
    ):
        flow = yaml.safe_load((directory / filename).read_text())
        simulator = _node(flow, "simulator")
        policy = _node(flow, "policy")
        assert simulator["path"] == "../fastwam_dora_support/project_python.sh"
        assert set(policy["inputs"]) == {
            "tick",
            "proprio_state",
            "policy_command",
            *image_ids,
        }
        assert policy["inputs"]["policy_command"] == "simulator/policy_command"
        assert set(policy["outputs"]) == {"action", "policy_command_status"}
        assert simulator["inputs"]["action"] == "policy/action"
        assert (
            simulator["inputs"]["policy_command_status"]
            == "policy/policy_command_status"
        )
        assert f"--config {policy_config}" in policy["args"]
