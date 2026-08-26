"""Forge SessionToolEndpoint implementation for a LeRobot policy runner."""

from __future__ import annotations

from collections.abc import Mapping

from forge_tool import (
    SessionToolEndpoint,
    ToolAccepted,
    ToolContext,
    ToolControlResponse,
    ToolEndpointError,
    ToolError,
    ToolEventEmitter,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolRequest,
    ToolResult,
    ToolResultResponse,
)

from .descriptor import LEROBOT_ENDPOINT_DESCRIPTOR
from .lifecycle import SessionCommandError, SessionLifecycleAdapter


class LeRobotPolicySessionEndpoint(SessionToolEndpoint):
    """Adapt Forge Session lifecycle calls to a policy-command adapter.

    This class implements the upstream protocol contract but does not provide a
    second transport.  Until the upstream Session dispatcher is available, the
    ``DoraToolEndpointBinding`` cannot dispatch this descriptor; the separate
    command binding below remains usable and testable today.
    """

    descriptor = LEROBOT_ENDPOINT_DESCRIPTOR

    def __init__(self, adapter: SessionLifecycleAdapter) -> None:
        if not isinstance(adapter, SessionLifecycleAdapter):
            raise TypeError("adapter must be a SessionLifecycleAdapter")
        self.adapter = adapter
        self._results: dict[str, ToolResult] = {}

    async def start(
        self,
        request: ToolRequest,
        context: ToolContext,
        events: ToolEventEmitter,
    ) -> ToolAccepted:
        if not isinstance(request, ToolRequest):
            raise TypeError("request must be a ToolRequest")
        if not isinstance(context, ToolContext):
            raise TypeError("context must be a ToolContext")
        instruction = _instruction_from_request(request.arguments)
        try:
            admission = self.adapter.start(
                session_id=context.invocation_id,
                instruction=instruction,
            )
        except SessionCommandError as exc:
            raise ToolEndpointError(
                ToolError(
                    code="POLICY_SESSION_COMMAND_FAILED",
                    message=str(exc),
                    retryable=True,
                )
            ) from exc
        if not admission.accepted:
            raise ToolEndpointError(
                ToolError(
                    code="POLICY_SESSION_BUSY",
                    message=admission.reason or "another policy session is active",
                    retryable=True,
                    details={"active_session_id": self.adapter.active_session_id},
                )
            )
        return ToolAccepted(
            details={
                "session_id": admission.session_id,
                "state": admission.state,
                "duplicate": admission.duplicate,
            }
        )

    async def stop(
        self,
        key: ToolExecutionKey,
        reason: str | None = None,
    ) -> ToolControlResponse:
        if not isinstance(key, ToolExecutionKey):
            raise TypeError("key must be a ToolExecutionKey")
        control = self.adapter.stop(session_id=key.invocation_id, reason=reason)
        if control.accepted:
            return ToolControlResponse(
                command="stop",
                status="terminal" if control.terminal else "accepted",
                details={"session_id": control.session_id, "state": control.state},
            )
        return ToolControlResponse(
            command="stop",
            status="rejected",
            error=ToolError(
                code="POLICY_SESSION_NOT_ACTIVE",
                message=control.reason or "session is not active",
                retryable=False,
            ),
        )

    async def status(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        if not isinstance(key, ToolExecutionKey):
            raise TypeError("key must be a ToolExecutionKey")
        snapshot = self.adapter.snapshot()
        if snapshot.session_id != key.invocation_id:
            return ToolExecutionStatus(
                phase="unknown",
                error=ToolError(
                    code="POLICY_SESSION_NOT_FOUND",
                    message="session is not known by this endpoint",
                    retryable=False,
                ),
            )
        if snapshot.state == "idle":
            return ToolExecutionStatus(
                phase="unknown",
                error=ToolError(
                    code="POLICY_SESSION_NOT_FOUND",
                    message="session has not been admitted",
                    retryable=False,
                ),
            )
        if snapshot.state == "failed":
            return ToolExecutionStatus(
                phase="failed",
                error=ToolError(
                    code="POLICY_RUNNER_ERROR",
                    message=snapshot.last_error or "runner failed",
                    retryable=False,
                ),
            )
        return ToolExecutionStatus(phase=snapshot.state)  # type: ignore[arg-type]

    async def result(self, key: ToolExecutionKey) -> ToolResultResponse:
        cached = self._results.get(key.invocation_id)
        if cached is not None:
            return ToolResultResponse(status="available", result=cached)
        status = await self.status(key)
        if status.phase in ("accepted", "running", "stopping"):
            return ToolResultResponse(status="pending")
        if status.phase == "failed":
            result = ToolResult(
                status="failed",
                error=status.error,
            )
        elif status.phase == "stopped":
            result = ToolResult(
                status="stopped",
                outputs={"session_id": key.invocation_id},
            )
        else:
            return ToolResultResponse(status="not_found")
        self._results[key.invocation_id] = result
        return ToolResultResponse(status="available", result=result)


def _instruction_from_request(arguments: Mapping[str, object]) -> str | None:
    unexpected = set(arguments) - {"instruction"}
    if unexpected:
        raise ToolEndpointError(
            ToolError(
                code="POLICY_SESSION_INVALID_ARGUMENT",
                message="execute accepts only the optional instruction argument",
                details={"unexpected": sorted(unexpected)},
            )
        )
    instruction = arguments.get("instruction")
    if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
        raise ToolEndpointError(
            ToolError(
                code="POLICY_SESSION_INVALID_ARGUMENT",
                message="instruction must be a non-empty string",
            )
        )
    return instruction


__all__ = ["LeRobotPolicySessionEndpoint"]
