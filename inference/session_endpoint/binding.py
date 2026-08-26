"""Dora control-topic binding for the LeRobot Session adapter."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from forge_msgs import PolicyCommand, PolicyCommandStatus

from .endpoint import LeRobotPolicySessionEndpoint
from .lifecycle import SessionLifecycleAdapter

logger = logging.getLogger(__name__)


class LeRobotPolicyCommandBinding:
    """Connect lifecycle commands to the existing policy Dora topics.

    The binding deliberately handles only ``policy_command_status`` input and
    publishes only ``policy_command`` output.  It does not subscribe to or
    forward ``image/*``, ``proprio_state``, ``tick``, or ``action``.
    """

    def __init__(
        self,
        node: Any,
        *,
        policy_id: str,
        command_output_id: str = "policy_command",
        status_input_id: str = "policy_command_status",
    ) -> None:
        send_output = getattr(node, "send_output", None)
        if not callable(send_output):
            raise TypeError("node must provide send_output(output_id, value)")
        self.node = node
        self.command_output_id = command_output_id
        self.status_input_id = status_input_id
        self.adapter = SessionLifecycleAdapter(
            policy_id=policy_id, send_command=self._send_command
        )
        # This is the protocol implementation to hand to the future Gateway
        # Session dispatcher.  It is intentionally not dispatched here while
        # upstream Session transport support is unavailable.
        self.endpoint = LeRobotPolicySessionEndpoint(self.adapter)

    def _send_command(self, command: PolicyCommand) -> None:
        self.node.send_output(self.command_output_id, command.to_arrow())

    def process_event(self, event: dict[str, Any]) -> int | None:
        """Process only control/status events from a Dora endpoint node."""

        event_type = event.get("type")
        if event_type == "STOP":
            self.adapter.on_shutdown()
            return 0
        if event_type == "ERROR":
            self.adapter.on_runner_error(str(event.get("error") or "runner node error"))
            return 1
        if event_type != "INPUT" or event.get("id") != self.status_input_id:
            return None
        value = event.get("value")
        if value is None:
            return None
        try:
            self.adapter.handle_status(PolicyCommandStatus.from_arrow(value))
        except (TypeError, ValueError) as exc:
            logger.warning("ignored invalid policy_command_status input: %s", exc)
        return None

    def run(self, events: Iterable[dict[str, Any]]) -> int:
        for event in events:
            result = self.process_event(event)
            if result is not None:
                return result
        return 0


__all__ = ["LeRobotPolicyCommandBinding"]
