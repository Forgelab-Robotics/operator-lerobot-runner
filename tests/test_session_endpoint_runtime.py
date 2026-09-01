from __future__ import annotations

import asyncio

import pytest
from forge_tool import ToolContext, ToolExecutionKey, ToolRequest

from lerobot_inference.inference.config import PolicyNodeConfig
from lerobot_inference.inference.session_endpoint import (
    DESCRIPTOR,
    LeRobotServeSessionEndpoint,
)


class _Emitter:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


def _context(invocation_id: str) -> ToolContext:
    return ToolContext(
        endpoint_id="policy.runner",
        tool_id="libero.policy",
        implementation_id="lerobot.lingbot_va",
        operation="serve",
        execution_key=ToolExecutionKey(invocation_id, "attempt-1"),
        caller_id="test",
        deadline_ms=None,
    )


def test_descriptor_is_stoppable_session() -> None:
    operation = DESCRIPTOR.operations[0]
    assert operation.name == "serve"
    assert operation.semantics == "session"
    assert operation.stoppable is True
    assert operation.status_supported is True
    assert operation.max_concurrency == 1


def test_config_accepts_session_mode_and_expands_model_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LINGBOT_VA_MODEL_DIR", str(tmp_path / "policy"))
    monkeypatch.setenv("LINGBOT_VA_BASE_DIR", str(tmp_path / "base"))
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        """
mode: session_endpoint
joints: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
policy:
  type: lingbot_va
  pretrained_path: ${LINGBOT_VA_MODEL_DIR}
  wan_pretrained_path: ${LINGBOT_VA_BASE_DIR}
image_inputs:
  image/main: image
  image/second: image2
""",
        encoding="utf-8",
    )
    config = PolicyNodeConfig.from_yaml_path(config_path)
    assert config.mode == "session_endpoint"
    assert config.policy["pretrained_path"] == str(tmp_path / "policy")
    assert config.policy["wan_pretrained_path"] == str(tmp_path / "base")


def test_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        PolicyNodeConfig.from_dict(
            {
                "mode": "other",
                "joints": ["joint1"],
                "policy": {"type": "act"},
                "image_inputs": {"image/main": "image"},
            }
        )


def test_session_lifecycle_and_stop_hook() -> None:
    endpoint = LeRobotServeSessionEndpoint(policy_id="lingbot_va_libero")
    stopped = []
    endpoint.bind_stop_hook(lambda: stopped.append(True))
    emitter = _Emitter()
    context = _context("session-1")

    accepted = asyncio.run(endpoint.start(ToolRequest(arguments={}), context, emitter))
    assert accepted.details["resident"] is True
    assert asyncio.run(endpoint.status(context.execution_key)).phase == "running"
    assert asyncio.run(endpoint.result(context.execution_key)).status == "pending"

    control = asyncio.run(endpoint.stop(context.execution_key, "done"))
    assert control.status == "accepted"
    assert stopped == [True]
    assert asyncio.run(endpoint.status(context.execution_key)).phase == "stopped"
    result = asyncio.run(endpoint.result(context.execution_key))
    assert result.status == "available"
    assert result.result is not None
    assert result.result.status == "stopped"
    assert result.result.outputs["policy_id"] == "lingbot_va_libero"
    assert [event.type for event in emitter.events] == ["stopped"]


def test_duplicate_is_idempotent_and_other_session_is_busy() -> None:
    endpoint = LeRobotServeSessionEndpoint(policy_id="lingbot_va_libero")
    first = _context("session-1")
    emitter = _Emitter()
    asyncio.run(endpoint.start(ToolRequest(arguments={}), first, emitter))
    duplicate = asyncio.run(endpoint.start(ToolRequest(arguments={}), first, emitter))
    assert duplicate.details["idempotent"] is True

    with pytest.raises(Exception) as caught:
        asyncio.run(
            endpoint.start(ToolRequest(arguments={}), _context("session-2"), _Emitter())
        )
    assert "FORGE_ENDPOINT_BUSY" in str(caught.value)
