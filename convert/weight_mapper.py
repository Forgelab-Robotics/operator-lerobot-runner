"""Map policy_train DETR-ACT weights to LeRobot ACT parameter names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DIRECT_KEY_MAP: dict[str, str] = {
    "action_head.weight": "model.action_head.weight",
    "action_head.bias": "model.action_head.bias",
    "query_embed.weight": "model.decoder_pos_embed.weight",
    "input_proj_robot_state.weight": "model.encoder_robot_state_input_proj.weight",
    "input_proj_robot_state.bias": "model.encoder_robot_state_input_proj.bias",
    "encoder_action_proj.weight": "model.vae_encoder_action_input_proj.weight",
    "encoder_action_proj.bias": "model.vae_encoder_action_input_proj.bias",
    "encoder_joint_proj.weight": "model.vae_encoder_robot_state_input_proj.weight",
    "encoder_joint_proj.bias": "model.vae_encoder_robot_state_input_proj.bias",
    "latent_proj.weight": "model.vae_encoder_latent_output_proj.weight",
    "latent_proj.bias": "model.vae_encoder_latent_output_proj.bias",
    "cls_embed.weight": "model.vae_encoder_cls_embed.weight",
    "latent_out_proj.weight": "model.encoder_latent_input_proj.weight",
    "latent_out_proj.bias": "model.encoder_latent_input_proj.bias",
    "additional_pos_embed.weight": "model.encoder_1d_feature_pos_embed.weight",
    "pos_table": "model.vae_encoder_pos_enc",
    "input_proj.weight": "model.encoder_img_feat_input_proj.weight",
    "input_proj.bias": "model.encoder_img_feat_input_proj.bias",
}

PREFIX_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("encoder.layers.", "model.vae_encoder.layers."),
    ("encoder.norm.", "model.vae_encoder.norm."),
    ("transformer.encoder.layers.", "model.encoder.layers."),
    ("transformer.encoder.norm.", "model.encoder.norm."),
    ("transformer.decoder.layers.", "model.decoder.layers."),
    ("transformer.decoder.norm.", "model.decoder.norm."),
    ("backbones.0.0.body.", "model.backbone."),
    ("backbones.0.body.", "model.backbone."),
)

SKIP_SOURCE_PREFIXES = (
    "is_pad_head.",
)


@dataclass
class WeightMapReport:
    mapped: dict[str, str]
    skipped: list[str]
    unmapped: list[str]


def _map_single_key(source_key: str) -> str | None:
    for prefix in SKIP_SOURCE_PREFIXES:
        if source_key.startswith(prefix):
            return None
    if source_key in DIRECT_KEY_MAP:
        return DIRECT_KEY_MAP[source_key]
    for src_prefix, dst_prefix in PREFIX_KEY_MAP:
        if source_key.startswith(src_prefix):
            return dst_prefix + source_key[len(src_prefix) :]
    return None


def map_policy_train_state_dict(source: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], WeightMapReport]:
    mapped: dict[str, torch.Tensor] = {}
    mapping: dict[str, str] = {}
    skipped: list[str] = []
    unmapped: list[str] = []

    for key, tensor in source.items():
        target_key = _map_single_key(key)
        if target_key is None:
            if any(key.startswith(prefix) for prefix in SKIP_SOURCE_PREFIXES):
                skipped.append(key)
            else:
                unmapped.append(key)
            continue
        mapped[target_key] = tensor
        mapping[key] = target_key

    report = WeightMapReport(mapped=mapping, skipped=skipped, unmapped=unmapped)
    return mapped, report


def shape_match_report(
    mapped: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    missing_in_target: list[str] = []
    for key, tensor in mapped.items():
        if key not in target_state_dict:
            missing_in_target.append(key)
            continue
        target_shape = tuple(target_state_dict[key].shape)
        source_shape = tuple(tensor.shape)
        if target_shape != source_shape:
            mismatches.append(
                {"key": key, "source_shape": source_shape, "target_shape": target_shape}
            )
    return {
        "missing_in_target": missing_in_target,
        "shape_mismatches": mismatches,
    }
