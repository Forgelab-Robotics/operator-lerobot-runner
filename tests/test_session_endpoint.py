from __future__ import annotations

import asyncio
import inspect

import pytest
from forge_msgs import PolicyCommandStatus
from forge_tool import (
    ToolContext,
    ToolEndpointError,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolRequest,
)

from lerobot_inference.inference.session_endpoint import (
    LEROBOT_ENDPOINT_DESCRIPTOR,
    LEROBOT_ENDPOINT_ID,
    LEROBOT_OPERATION,
    LEROBOT_TOOL_ID,
    LeRobotPolicyCommandBinding,
    LeRobotPolicySessionEndpoint,
    SessionLifecycleAdapter,
)


def _status(*, policy_id: str, command: str, request_id: str, state: str, message: str = ""):
    return PolicyCommandStatus.from_outputs(
        policy_id=policy_id,
        command=command,
        request_id=request_id,
        status=state,
        message=message,
    )


def test_descriptor_is_a_single_formal_session_operation():
    assert LEROBOT_TOOL_ID == "policy.lerobot/execute"
    assert LEROBOT_ENDPOINT_ID == "policy.lerobot"
    assert LEROBOT_OPERATION == "execute"
    operation = LEROBOT_ENDPOINT_DESCRIPTOR.operations[0]
    assert operation.name == "execute"
    assert operation.semantics == "session"
    assert operation.stoppable is True
    assert operation.status_supported is True
    assert operation.max_concurrency == 1


def test_start_with_instruction_publishes_set_instruction_before_start():
    commands = []
    adapter = SessionLifecycleAdapter(
        policy_id="libero",
        send_command=commands.append,
    )

    admission = adapter.start(session_id="session-a", instruction="open the drawer")

    assert admission.accepted
    assert adapter.state == "accepted"
    assert [(item.command, item.request_id, item.inputs()) for item in commands] == [
        ("set_instruction", "session-a:set_instruction", {"instruction": "open the drawer"}),
        ("start", "session-a:start", {}),
    ]
    assert adapter.handle_status(
        _status(
            policy_id="libero",
            command="set_instruction",
            request_id="session-a:set_instruction",
            state="done",
        )
    )
    assert adapter.handle_status(
        _status(
            policy_id="libero",
            command="start",
            request_id="session-a:start",
            state="done",
        )
    )
    assert adapter.state == "running"


def test_start_without_instruction_only_publishes_start():
    commands = []
    adapter = SessionLifecycleAdapter(policy_id="libero", send_command=commands.append)

    adapter.start(session_id="session-a")

    assert [item.command for item in commands] == ["start"]


def test_stop_publishes_stop_and_done_status_stops_session():
    commands = []
    adapter = SessionLifecycleAdapter(policy_id="libero", send_command=commands.append)
    adapter.start(session_id="session-a")
    adapter.handle_status(
        _status(policy_id="libero", command="start", request_id="session-a:start", state="done")
    )

    control = adapter.stop(session_id="session-a", reason="operator requested stop")

    assert control.accepted
    assert adapter.state == "stopping"
    assert commands[-1].command == "stop"
    assert commands[-1].request_id == "session-a:stop"
    assert commands[-1].inputs() == {"reason": "operator requested stop"}
    adapter.handle_status(
        _status(policy_id="libero", command="stop", request_id="session-a:stop", state="done")
    )
    assert adapter.state == "stopped"


def test_late_start_ack_does_not_resurrect_a_stopping_session():
    adapter = SessionLifecycleAdapter(policy_id="libero", send_command=lambda command: None)
    adapter.start(session_id="session-a")
    adapter.stop(session_id="session-a")

    adapter.handle_status(
        _status(policy_id="libero", command="start", request_id="session-a:start", state="done")
    )
    assert adapter.state == "stopping"
    adapter.handle_status(
        _status(policy_id="libero", command="stop", request_id="session-a:stop", state="done")
    )
    assert adapter.state == "stopped"


def test_process_shutdown_stops_only_a_live_session():
    adapter = SessionLifecycleAdapter(policy_id="libero", send_command=lambda command: None)
    assert not adapter.on_shutdown()
    adapter.start(session_id="session-a")
    assert adapter.on_shutdown()
    assert adapter.state == "stopped"
    assert not adapter.on_shutdown()


def test_busy_second_session_is_rejected_without_a_second_command():
    commands = []
    adapter = SessionLifecycleAdapter(policy_id="libero", send_command=commands.append)
    adapter.start(session_id="session-a")

    admission = adapter.start(session_id="session-b")

    assert not admission.accepted
    assert admission.reason == "another policy session is already active"
    assert [item.request_id for item in commands] == ["session-a:start"]


def test_runner_error_status_fails_only_the_correlated_session():
    commands = []
    adapter = SessionLifecycleAdapter(policy_id="libero", send_command=commands.append)
    adapter.start(session_id="session-a")

    assert adapter.handle_status(
        _status(
            policy_id="libero",
            command="start",
            request_id="session-a:start",
            state="error",
            message="policy crashed",
        )
    )
    assert adapter.state == "failed"
    assert adapter.snapshot().last_error == "policy crashed"
    assert not adapter.handle_status(
        _status(
            policy_id="other-policy",
            command="start",
            request_id="session-a:start",
            state="done",
        )
    )


def test_binding_only_connects_policy_control_topics():
    class FakeNode:
        def __init__(self):
            self.outputs = []

        def send_output(self, output_id, value):
            self.outputs.append((output_id, value))

    node = FakeNode()
    binding = LeRobotPolicyCommandBinding(node, policy_id="libero")
    binding.adapter.start(session_id="session-a")
    assert [output_id for output_id, _ in node.outputs] == ["policy_command"]

    command = node.outputs[0][1]
    binding.process_event(
        {
            "type": "INPUT",
            "id": "policy_command_status",
            "value": _status(
                policy_id="libero",
                command="start",
                request_id="session-a:start",
                state="done",
            ).to_arrow(),
        }
    )
    assert binding.adapter.state == "running"
    # No observation/action data-plane IDs are consumed or emitted here.
    binding.process_event({"type": "INPUT", "id": "image/top", "value": object()})
    binding.process_event({"type": "INPUT", "id": "proprio_state", "value": object()})
    binding.process_event({"type": "INPUT", "id": "tick", "value": object()})
    assert len(node.outputs) == 1
    assert command is not None


def test_formal_session_endpoint_maps_lifecycle_to_forge_status_and_result():
    commands = []
    adapter = SessionLifecycleAdapter(policy_id="libero", send_command=commands.append)
    endpoint = LeRobotPolicySessionEndpoint(adapter)
    context = ToolContext(
        execution_key=ToolExecutionKey(invocation_id="session-a", attempt_id="attempt-1"),
        tool_id=LEROBOT_TOOL_ID,
        implementation_id="lerobot-instance",
        endpoint_id=LEROBOT_ENDPOINT_ID,
        operation=LEROBOT_OPERATION,
    )

    async def exercise():
        accepted = await endpoint.start(ToolRequest({"instruction": "pick"}), context, None)
        assert accepted.details["session_id"] == "session-a"
        adapter.handle_status(
            _status(policy_id="libero", command="start", request_id="session-a:start", state="done")
        )
        status = await endpoint.status(context.execution_key)
        assert isinstance(status, ToolExecutionStatus)
        assert status.phase == "running"
        control = await endpoint.stop(context.execution_key, "done")
        assert control.status == "accepted"
        adapter.handle_status(
            _status(policy_id="libero", command="stop", request_id="session-a:stop", state="done")
        )
        result = await endpoint.result(context.execution_key)
        assert result.status == "available"
        assert result.result.status == "stopped"

    asyncio.run(exercise())


def test_endpoint_rejects_unexpected_arguments_and_busy_invocation():
    adapter = SessionLifecycleAdapter(policy_id="libero", send_command=lambda command: None)
    endpoint = LeRobotPolicySessionEndpoint(adapter)
    context = ToolContext(
        execution_key=ToolExecutionKey(invocation_id="session-a", attempt_id="attempt-1"),
        tool_id=LEROBOT_TOOL_ID,
        implementation_id="lerobot-instance",
        endpoint_id=LEROBOT_ENDPOINT_ID,
        operation=LEROBOT_OPERATION,
    )

    async def exercise():
        with pytest.raises(ToolEndpointError) as invalid:
            await endpoint.start(ToolRequest({"task": "pick"}), context, None)
        assert invalid.value.error.code == "POLICY_SESSION_INVALID_ARGUMENT"
        await endpoint.start(ToolRequest({}), context, None)
        other_context = ToolContext(
            execution_key=ToolExecutionKey(invocation_id="session-b", attempt_id="attempt-1"),
            tool_id=LEROBOT_TOOL_ID,
            implementation_id="lerobot-instance",
            endpoint_id=LEROBOT_ENDPOINT_ID,
            operation=LEROBOT_OPERATION,
        )
        with pytest.raises(ToolEndpointError) as busy:
            await endpoint.start(ToolRequest({}), other_context, None)
        assert busy.value.error.code == "POLICY_SESSION_BUSY"

    asyncio.run(exercise())


def test_endpoint_modules_do_not_own_data_plane_forwarding():
    source = inspect.getsource(LeRobotPolicyCommandBinding)
    assert "send_output" in source
    assert "send_input" not in source
    assert "generate_action" not in source
