"""Load LeRobot policy + preprocessor/postprocessor from a pretrained_model directory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

logger = logging.getLogger(__name__)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_policy_bundle(
    pretrained_path: str | Path,
    *,
    device: str | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
) -> tuple[PreTrainedPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline, PreTrainedConfig]:
    """Load policy weights and processor pipelines from a local pretrained_model directory."""
    path = Path(pretrained_path).expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(path)
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(path)

    dev = resolve_device(device or getattr(config, "device", None))
    policy.to(dev)
    policy.eval()

    overrides = dict(preprocessor_overrides or {})
    overrides.setdefault("device_processor", {"device": str(dev)})

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(path),
        preprocessor_overrides=overrides,
    )
    logger.info(
        "Loaded LeRobot policy type=%s from %s on %s",
        config.type,
        path,
        dev,
    )
    return policy, preprocessor, postprocessor, config
