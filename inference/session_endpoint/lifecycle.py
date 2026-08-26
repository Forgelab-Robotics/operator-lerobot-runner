"""Policy-command backed lifecycle state for one LeRobot execution session.

This module intentionally knows nothing about images, proprioception, ticks,
actions, or task success.  It only translates a Session lifecycle into the
runner's existing ``policy_command`` control messages and consumes their
``policy_command_status`` acknowledgements.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import uuid
from typing import Literal

from forge_msgs import PolicyCommand, PolicyCommandStatus

SessionState = Literal[
    "idle",
    "accepted",
    "running",
    "stopping",
    "failed",
    "stopped",
]

_ACTIVE_STATES = frozenset(("accepted", "running", "stopping"))
_TERMINAL_STATES = frozenset(("failed", "stopped"))


@dataclass(frozen=True)
class SessionAdmission:
    """Result of attempting to admit one Session invocation."""

    accepted: bool
    session_id: str
    state: SessionState
    reason: str = ""
    duplicate: bool = False


@dataclass(frozen=True)
class SessionControl:
    """Result of attempting to stop one Session invocation."""

    accepted: bool
    session_id: str
    state: SessionState
    reason: str = ""
    terminal: bool = False


@dataclass(frozen=True)
class SessionSnapshot:
    """Immutable view of the adapter's control-plane state."""

    session_id: str | None
    state: SessionState
    instruction: str | None = None
    last_error: str = ""
    pending_commands: tuple[str, ...] = field(default_factory=tuple)


class SessionCommandError(RuntimeError):
    """The runner control topic rejected a command before it was published."""


class SessionLifecycleAdapter:
    """Translate Session lifecycle calls to existing ``PolicyCommand`` values.

    ``send_command`` is injected so the adapter remains deterministic and
    unit-testable.  The production Dora binding supplies a callback that calls
    ``Node.send_output("policy_command", command.to_arrow())``.
    """

    def __init__(
        self,
        *,
        policy_id: str,
        send_command: Callable[[PolicyCommand], None],
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if not callable(send_command):
            raise TypeError("send_command must be callable")
        self.policy_id = policy_id
        self._send_command = send_command
        self._session_id_factory = session_id_factory or (lambda: str(uuid.uuid4()))
        self._active_session_id: str | None = None
        self._state: SessionState = "idle"
        self._instruction: str | None = None
        self._last_error = ""
        self._pending_commands: dict[str, str] = {}

    @property
    def active_session_id(self) -> str | None:
        return self._active_session_id

    @property
    def state(self) -> SessionState:
        return self._state

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self._active_session_id,
            state=self._state,
            instruction=self._instruction,
            last_error=self._last_error,
            pending_commands=tuple(self._pending_commands),
        )

    def start(
        self,
        *,
        session_id: str | None = None,
        instruction: str | None = None,
    ) -> SessionAdmission:
        """Admit one session and publish ``set_instruction`` then ``start``.

        A second invocation cannot run concurrently because the runner owns one
        policy instance.  Replaying the same invocation is idempotent while it
        is active; another invocation is rejected as busy.
        """

        requested_id = session_id or self._session_id_factory()
        _require_session_id(requested_id)
        if self._state in _ACTIVE_STATES:
            if requested_id == self._active_session_id:
                return SessionAdmission(
                    accepted=True,
                    session_id=requested_id,
                    state=self._state,
                    reason="session already active",
                    duplicate=True,
                )
            return SessionAdmission(
                accepted=False,
                session_id=requested_id,
                state=self._state,
                reason="another policy session is already active",
            )

        # A terminal/idle record can be replaced by the next invocation.  The
        # active session identity is intentionally local because PolicyCommand
        # has no invocation_id/session_id field in the shared schema.
        self._active_session_id = requested_id
        self._state = "accepted"
        self._instruction = instruction
        self._last_error = ""
        self._pending_commands.clear()

        try:
            if instruction is not None:
                _require_instruction(instruction)
                self._publish(
                    command="set_instruction",
                    request_id=f"{requested_id}:set_instruction",
                    inputs={"instruction": instruction},
                )
            self._publish(
                command="start",
                request_id=f"{requested_id}:start",
                inputs={},
            )
        except Exception as exc:
            self._state = "failed"
            self._last_error = str(exc)
            raise SessionCommandError(self._last_error) from exc

        return SessionAdmission(
            accepted=True,
            session_id=requested_id,
            state=self._state,
        )

    def stop(self, *, session_id: str | None = None, reason: str | None = None) -> SessionControl:
        """Publish the runner's existing ``stop`` command for this session."""

        requested_id = session_id or self._active_session_id or ""
        _require_session_id(requested_id)
        if self._active_session_id != requested_id:
            return SessionControl(
                accepted=False,
                session_id=requested_id,
                state=self._state,
                reason="session is not active on this endpoint",
            )
        if self._state in _TERMINAL_STATES:
            return SessionControl(
                accepted=True,
                session_id=requested_id,
                state=self._state,
                reason="session is already terminal",
                terminal=True,
            )
        if self._state == "stopping":
            return SessionControl(
                accepted=True,
                session_id=requested_id,
                state=self._state,
                reason="stop already requested",
            )

        self._state = "stopping"
        try:
            inputs = {"reason": reason} if reason else {}
            self._publish(
                command="stop",
                request_id=f"{requested_id}:stop",
                inputs=inputs,
            )
        except Exception as exc:
            self._state = "failed"
            self._last_error = str(exc)
            return SessionControl(
                accepted=False,
                session_id=requested_id,
                state=self._state,
                reason=self._last_error,
            )
        return SessionControl(
            accepted=True,
            session_id=requested_id,
            state=self._state,
        )

    def handle_status(self, status: PolicyCommandStatus) -> bool:
        """Apply one correlated runner acknowledgement.

        Returns ``True`` only when the status belongs to the active session.
        Status from another policy or an unknown request is ignored, preventing
        stale messages from changing a newer session.
        """

        if not isinstance(status, PolicyCommandStatus):
            raise TypeError("status must be a PolicyCommandStatus")
        if status.policy_id != self.policy_id:
            return False
        request_id = status.request_id
        command = self._pending_commands.get(request_id)
        if command is None or self._active_session_id is None:
            return False
        expected_id = f"{self._active_session_id}:{command}"
        if request_id != expected_id:
            return False
        if status.command != command:
            self._fail("POLICY_SESSION_CORRELATION_ERROR: command does not match request_id")
            self._pending_commands.pop(request_id, None)
            return True

        if status.status in ("error", "rejected"):
            self._fail(status.message or f"runner rejected {command}")
            self._pending_commands.pop(request_id, None)
            return True

        if command == "stop":
            if status.status == "done":
                if self._state in _ACTIVE_STATES:
                    self._state = "stopped"
                self._pending_commands.pop(request_id, None)
            elif status.status in ("accepted", "running"):
                self._state = "stopping"
            return True

        if command == "start":
            if status.status == "done":
                # A stop may legitimately race the start acknowledgement.  Do
                # not resurrect a session already entering its terminal path.
                if self._state == "accepted":
                    self._state = "running"
                self._pending_commands.pop(request_id, None)
            elif status.status in ("accepted", "running"):
                # Keep the admission state until the authoritative done ack;
                # a stop/failure racing this ack must remain terminal-path.
                pass
            return True

        # set_instruction is complete when the runner reports done; the Session
        # remains accepted until the separate start acknowledgement arrives.
        if status.status == "done":
            self._pending_commands.pop(request_id, None)
        return True

    def on_runner_error(self, message: str) -> bool:
        """Mark the active session failed after a fatal runner error."""

        if self._active_session_id is None or self._state not in _ACTIVE_STATES:
            return False
        self._fail(message or "runner failed")
        return True

    def on_shutdown(self) -> bool:
        """Mark a still-active session stopped when the process exits."""

        if self._active_session_id is None or self._state not in _ACTIVE_STATES:
            return False
        self._state = "stopped"
        self._pending_commands.clear()
        return True

    def _publish(self, *, command: str, request_id: str, inputs: Mapping[str, object]) -> None:
        self._pending_commands[request_id] = command
        payload = PolicyCommand.from_inputs(
            policy_id=self.policy_id,
            command=command,
            request_id=request_id,
            inputs=dict(inputs),
        )
        self._send_command(payload)

    def _fail(self, message: str) -> None:
        self._state = "failed"
        self._last_error = message


def _require_session_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("session_id must be a non-empty string")


def _require_instruction(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("instruction must be a non-empty string when provided")


__all__ = [
    "SessionAdmission",
    "SessionCommandError",
    "SessionControl",
    "SessionLifecycleAdapter",
    "SessionSnapshot",
]
