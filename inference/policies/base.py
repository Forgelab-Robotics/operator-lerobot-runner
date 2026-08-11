"""Policy adapter protocol for Dora inference nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LerobotPolicyAdapter(ABC):
    """Algorithm-specific wrapper around a LeRobot PreTrainedPolicy."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal action queues / ensemblers."""

    @abstractmethod
    def is_observation_needed(self) -> bool:
        """Return True when the next step requires a fresh observation."""

    @property
    def required_image_keys(self) -> frozenset[str] | None:
        """Image observation keys required by this loaded policy.

        ``None`` keeps all image inputs configured by the runtime. Adapters that
        resolve checkpoint features should return the exact active key set so the
        Dora runner can ignore unrelated image topics.
        """
        return None

    @abstractmethod
    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray | None:
        """Return one action vector, or None when an async backend is not ready."""
