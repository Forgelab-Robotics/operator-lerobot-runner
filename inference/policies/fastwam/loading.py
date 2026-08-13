"""Strict, local-only FastWAM policy loading."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

_WAN_SOURCE_LOCK = Lock()


@contextmanager
def _local_wan_vae_source(path: str | Path) -> Iterator[None]:
    """Point LeRobot 0.6.1's fixed Wan VAE source at a validated local snapshot."""
    from lerobot.policies.fastwam.wan import components

    local_path = str(Path(path).expanduser().resolve())
    with _WAN_SOURCE_LOCK:
        original = components.WAN22_DIFFUSERS_MODEL_ID
        components.WAN22_DIFFUSERS_MODEL_ID = local_path
        try:
            yield
        finally:
            components.WAN22_DIFFUSERS_MODEL_ID = original


def load_fastwam_policy_strict(
    policy_cls: type[PreTrainedPolicy],
    path: Path,
    config: PreTrainedConfig,
    *,
    wan_diffusers_path: str | Path,
) -> PreTrainedPolicy:
    """Load every policy tensor exactly; never permit cross-embodiment reinitialization."""
    with _local_wan_vae_source(wan_diffusers_path):
        return policy_cls.from_pretrained(path, config=config, strict=True)
