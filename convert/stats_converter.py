"""Convert policy_train dataset_stats.pkl to LeRobot dataset stats."""

from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pkl_to_lerobot_stats(
    stats: dict[str, np.ndarray],
    *,
    camera_names: list[str],
    image_shape: tuple[int, int, int] = (3, 480, 640),
) -> dict[str, dict[str, np.ndarray]]:
    state_mean = np.asarray(stats["qpos_mean"], dtype=np.float32).reshape(-1)
    state_std = np.asarray(stats["qpos_std"], dtype=np.float32).reshape(-1)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32).reshape(-1)
    action_std = np.asarray(stats["action_std"], dtype=np.float32).reshape(-1)

    _, height, width = image_shape
    mean_hwc = np.tile(IMAGENET_MEAN.reshape(1, 1, 3), (height, width, 1))
    std_hwc = np.tile(IMAGENET_STD.reshape(1, 1, 3), (height, width, 1))
    mean_chw = np.ascontiguousarray(mean_hwc.transpose(2, 0, 1))
    std_chw = np.ascontiguousarray(std_hwc.transpose(2, 0, 1))

    lerobot_stats: dict[str, dict[str, np.ndarray]] = {
        "observation.state": {"mean": state_mean, "std": state_std},
        "action": {"mean": action_mean, "std": action_std},
    }
    for camera in camera_names:
        lerobot_stats[camera] = {"mean": mean_chw, "std": std_chw}
    return lerobot_stats
