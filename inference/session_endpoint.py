"""Forge Session endpoint for an in-process LeRobot policy node.

The model is loaded before this endpoint registers, so an accepted session means
the policy process is resident and ready for benchmark data-plane commands.
Benchmark-owned ``PolicyCommand`` values still delimit episodes; this endpoint
only owns the longer-lived policy service lease.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from forge_tool import (
    TOOL_ENDPOINT_PROTOCOL,
    ToolAccepted,
    ToolContext,
    ToolControlResponse,
    ToolEndpointDescriptor,
    ToolEndpointError,
    ToolError,
    ToolEvent,
    ToolEventEmitter,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolOperationDescriptor,
    ToolRequest,
    ToolResult,
    ToolResultResponse,
)

ENDPOINT_ID = "policy.runner"
OPERATION = "serve"
DESCRIPTOR = ToolEndpointDescriptor(
    protocol_version=TOOL_ENDPOINT_PROTOCOL,
    endpoint_id=ENDPOINT_ID,
    operations=(
        ToolOperationDescriptor(
            name=OPERATION,
            semantics="session",
            stoppable=True,
            status_supported=True,
            max_concurrency=1,
        ),
    ),
)


def _endpoint_error(code: str, message: str, *, retryable: bool = False) -> ToolEndpointError:
    return ToolEndpointError(ToolError(code=code, message=message, retryable=retryable))


@dataclass(slots=True)
class _Execution:
    key: ToolExecutionKey
    arguments_json: str
    emitter: ToolEventEmitter
    phase: str = "running"
    result: ToolResult | None = None


class LeRobotServeSessionEndpoint:
    """One resident in-process policy service, reusable across benchmark runs."""

    def __init__(self, *, policy_id: str) -> None:
        if not policy_id:
            raise ValueError("policy_id must not be empty")
        self.policy_id = policy_id
        self._executions: dict[ToolExecutionKey, _Execution] = {}
        self._active: ToolExecutionKey | None = None
        self._accepting = True
        self._stop_hook: Callable[[], None] | None = None

    def bind_stop_hook(self, hook: Callable[[], None]) -> None:
        self._stop_hook = hook

    async def start(
        self,
        request: ToolRequest,
        context: ToolContext,
        events: ToolEventEmitter,
    ) -> ToolAccepted:
        if context.operation != OPERATION:
            raise _endpoint_error(
                "FORGE_PROTOCOL_UNKNOWN_OPERATION",
                f"unknown operation {context.operation!r}",
            )
        if not self._accepting:
            raise _endpoint_error(
                "FORGE_ENDPOINT_STOPPING",
                "policy.runner is shutting down",
                retryable=True,
            )
        if request.arguments:
            raise _endpoint_error(
                "POLICY_RUNNER_INVALID_ARGUMENTS",
                "policy.runner/serve accepts no arguments",
            )
        key = context.execution_key
        arguments_json = json.dumps(dict(request.arguments), sort_keys=True, separators=(",", ":"))
        existing = self._executions.get(key)
        if existing is not None:
            if existing.arguments_json != arguments_json:
                raise _endpoint_error(
                    "FORGE_EXECUTION_CONFLICT",
                    "execution key was reused with different arguments",
                )
            return ToolAccepted(details=self._details(existing, idempotent=True))
        if self._active is not None:
            raise _endpoint_error(
                "FORGE_ENDPOINT_BUSY",
                f"policy session {self._active.invocation_id!r} is already active",
                retryable=True,
            )
        execution = _Execution(
            key=key,
            arguments_json=arguments_json,
            emitter=events,
        )
        self._executions[key] = execution
        self._active = key
        return ToolAccepted(details=self._details(execution, idempotent=False))

    async def stop(
        self,
        key: ToolExecutionKey,
        reason: str | None = None,
    ) -> ToolControlResponse:
        execution = self._executions.get(key)
        if execution is None:
            return ToolControlResponse(
                command="stop",
                status="rejected",
                error=ToolError(
                    code="FORGE_EXECUTION_NOT_FOUND",
                    message="policy session is unknown",
                ),
            )
        if execution.result is not None:
            return ToolControlResponse(command="stop", status="terminal")
        execution.phase = "stopping"
        try:
            if self._stop_hook is not None:
                self._stop_hook()
        except Exception as exc:
            execution.phase = "failed"
            execution.result = ToolResult(
                status="failed",
                error=ToolError(
                    code="POLICY_RUNNER_STOP_FAILED",
                    message=str(exc),
                ),
            )
            if self._active == key:
                self._active = None
            return ToolControlResponse(
                command="stop",
                status="rejected",
                error=execution.result.error,
            )
        execution.phase = "stopped"
        execution.result = ToolResult(
            status="stopped",
            outputs={
                "status": "stopped",
                "policy_id": self.policy_id,
                "reason": reason,
            },
        )
        if self._active == key:
            self._active = None
        await execution.emitter.emit(
            ToolEvent(
                type="stopped",
                data={"policy_id": self.policy_id, "reason": reason},
            )
        )
        return ToolControlResponse(
            command="stop",
            status="accepted",
            details={"policy_id": self.policy_id, "reason": reason},
        )

    async def status(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        execution = self._executions.get(key)
        if execution is None:
            return ToolExecutionStatus(
                phase="unknown",
                error=ToolError(
                    code="FORGE_EXECUTION_NOT_FOUND",
                    message="policy session is unknown",
                ),
            )
        return ToolExecutionStatus(
            phase=execution.phase,  # type: ignore[arg-type]
            details=self._details(execution),
        )

    async def result(self, key: ToolExecutionKey) -> ToolResultResponse:
        execution = self._executions.get(key)
        if execution is None:
            return ToolResultResponse(status="not_found")
        if execution.result is None:
            return ToolResultResponse(status="pending")
        return ToolResultResponse(status="available", result=execution.result)

    async def shutdown(self) -> None:
        self._accepting = False
        if self._active is None:
            return
        execution = self._executions.get(self._active)
        if execution is None or execution.result is not None:
            return
        if self._stop_hook is not None:
            self._stop_hook()
        execution.phase = "stopped"
        execution.result = ToolResult(
            status="stopped",
            outputs={"status": "stopped", "policy_id": self.policy_id, "reason": "node shutdown"},
        )
        self._active = None

    def _details(self, execution: _Execution, idempotent: bool | None = None) -> dict[str, Any]:
        details: dict[str, Any] = {
            "policy_id": self.policy_id,
            "phase": execution.phase,
            "resident": True,
        }
        if idempotent is not None:
            details["idempotent"] = idempotent
        return details


__all__ = ["DESCRIPTOR", "ENDPOINT_ID", "LeRobotServeSessionEndpoint", "OPERATION"]
