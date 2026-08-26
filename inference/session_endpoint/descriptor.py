"""Formal Forge descriptor for the LeRobot policy session endpoint."""

from forge_tool import (
    TOOL_ENDPOINT_PROTOCOL,
    ToolEndpointDescriptor,
    ToolOperationDescriptor,
)

LEROBOT_TOOL_ID = "policy.lerobot/execute"
LEROBOT_ENDPOINT_ID = "policy.lerobot"
LEROBOT_OPERATION = "execute"

# Keep this as a real Session descriptor even while the current upstream
# ToolEndpointHandler/Gateway transport is still query/action-only.
LEROBOT_ENDPOINT_DESCRIPTOR = ToolEndpointDescriptor(
    protocol_version=TOOL_ENDPOINT_PROTOCOL,
    endpoint_id=LEROBOT_ENDPOINT_ID,
    operations=(
        ToolOperationDescriptor(
            name=LEROBOT_OPERATION,
            semantics="session",
            stoppable=True,
            status_supported=True,
            max_concurrency=1,
        ),
    ),
)

# Short aliases follow the naming used by existing policy-node endpoint
# packages, while the explicit LEROBOT_* names remain the public constants.
ENDPOINT_ID = LEROBOT_ENDPOINT_ID
OPERATION = LEROBOT_OPERATION
DESCRIPTOR = LEROBOT_ENDPOINT_DESCRIPTOR
LEROBOT_DESCRIPTOR = LEROBOT_ENDPOINT_DESCRIPTOR

__all__ = [
    "LEROBOT_ENDPOINT_DESCRIPTOR",
    "LEROBOT_ENDPOINT_ID",
    "LEROBOT_OPERATION",
    "LEROBOT_TOOL_ID",
    "DESCRIPTOR",
    "ENDPOINT_ID",
    "LEROBOT_DESCRIPTOR",
    "OPERATION",
]
