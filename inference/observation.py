"""Convert forge/Dora observation dicts to LeRobot policy batches."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from einops import rearrange


def observation_to_policy_batch(observation: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Map HWC uint8 images and 1-D state vectors to batched CHW float tensors."""
    if not observation:
        raise ValueError("observation is empty; policy queue may be depleted incorrectly")

    batch: dict[str, torch.Tensor] = {}

    if "observation.state" in observation:
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        state_tensor = torch.from_numpy(state)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        batch["observation.state"] = state_tensor

    for key, value in observation.items():
        if not key.startswith("observation.images."):
            continue
        img = np.asarray(value)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        img_tensor = torch.from_numpy(img)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Expected HWC image for {key}, got shape {tuple(img_tensor.shape)}")
        img_tensor = rearrange(img_tensor, "b h w c -> b c h w").contiguous().float() / 255.0
        batch[key] = img_tensor

    return batch


def action_tensor_to_numpy(action: torch.Tensor) -> np.ndarray:
    """Convert policy output tensor to 1-D numpy action for JointCommand."""
    if action.dim() == 2:
        action = action[0]
    return action.detach().cpu().numpy().astype(np.float32, copy=False)
