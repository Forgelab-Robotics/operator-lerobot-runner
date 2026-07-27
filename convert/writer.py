"""Write converted LeRobot pretrained_model directory (flat under dst_dir)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from safetensors.torch import load_file

from lerobot_inference.convert.meta import PolicyTrainMeta
from lerobot_inference.convert.stats_converter import pkl_to_lerobot_stats
from lerobot_inference.convert.weight_mapper import map_policy_train_state_dict, shape_match_report


def load_source_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load trusted tensor-only policy_train weights without executing pickle code."""
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        source = load_file(str(path))
    elif suffix == ".ckpt":
        source = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported checkpoint format: {path.name}")

    if isinstance(source, dict) and "state_dict" in source:
        source = source["state_dict"]
    if not isinstance(source, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in source.items()
    ):
        raise TypeError(
            f"Checkpoint must contain a tensor-only state_dict, got {type(source).__name__}: {path}"
        )
    return dict(source)


def _build_act_config(meta: PolicyTrainMeta) -> ACTConfig:
    params = meta.policy_params
    chunk_size = int(params.get("chunk_size", params.get("num_queries", 100)))
    num_queries = int(params.get("num_queries", chunk_size))
    # 旧 policy_train ACT 预测 ``num_queries``（通常等于 chunk_size）个动作，
    # 但仅消费 ``query_size`` 个动作后就用新观测重新预测。LeRobot 的
    # n_action_steps 正是这个“实际消费后刷新观测”的数量，不能误设为
    # num_queries，否则会额外执行一整段陈旧动作。
    n_action_steps = int(params.get("query_size", num_queries))
    enc_layers = int(params.get("enc_layers", 4))

    input_features: dict[str, PolicyFeature] = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(meta.state_dim,))
    }
    for camera in meta.camera_names:
        input_features[camera] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))

    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(meta.action_dim,)),
    }

    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        dim_model=int(params.get("hidden_dim", 512)),
        n_heads=int(params.get("nheads", 8)),
        dim_feedforward=int(params.get("dim_feedforward", 3200)),
        n_encoder_layers=enc_layers,
        # policy_train 的 DETR 虽通常配置 dec_layers=7，但它在推理时
        # 对 transformer 输出取 ``[0]``，实际只使用第 0 个 decoder
        # layer。LeRobot 会依次执行全部配置层数，因此必须固定为 1，
        # 才能数值复现旧 pick_and_place ACT。
        n_decoder_layers=1,
        n_vae_encoder_layers=enc_layers,
        vision_backbone=str(params.get("backbone", "resnet18")),
        # Conversion loads a complete trained backbone from the source checkpoint;
        # avoid an unnecessary network download during model construction.
        pretrained_backbone_weights=None,
        kl_weight=float(params.get("kl_weight", 10.0)),
        optimizer_lr=float(params.get("lr", 2e-5)),
        optimizer_lr_backbone=float(params.get("lr_backbone", 2e-5)),
        use_vae=True,
    )


def convert_policy_train_to_lerobot(
    meta: PolicyTrainMeta,
    stats: dict[str, Any],
    dst_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write LeRobot loadable assets directly into ``dst_dir`` (no checkpoints/ nesting)."""
    dst_dir = dst_dir.expanduser().resolve()
    source_state = load_source_state_dict(meta.checkpoint_path)
    config = _build_act_config(meta)
    policy = ACTPolicy(config)
    mapped_state, map_report = map_policy_train_state_dict(source_state)
    # Compare against full policy.state_dict so keys keep the ``model.`` prefix.
    compatibility = shape_match_report(mapped_state, policy.state_dict())

    report: dict[str, Any] = {
        "source_checkpoint": str(meta.checkpoint_path),
        "source_stats": str(meta.stats_path),
        "destination": str(dst_dir),
        "mapped_keys": len(map_report.mapped),
        "skipped_keys": map_report.skipped,
        "unmapped_keys": map_report.unmapped,
        "compatibility": compatibility,
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    load_result = policy.load_state_dict(mapped_state, strict=False)
    report["missing_keys"] = load_result.missing_keys
    report["unexpected_keys"] = load_result.unexpected_keys

    dst_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(dst_dir)

    dataset_stats = pkl_to_lerobot_stats(stats, camera_names=meta.camera_names)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=dataset_stats,
    )
    preprocessor.save_pretrained(dst_dir, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(dst_dir, config_filename="policy_postprocessor.json")

    conversion_meta = {
        "source_checkpoint": meta.checkpoint_path.name,
        "source_task_id": meta.task_id,
        "camera_names": meta.camera_names,
        "policy_params": meta.policy_params,
        "map_report": {
            "mapped_count": len(map_report.mapped),
            "skipped": map_report.skipped,
            "unmapped": map_report.unmapped,
        },
    }
    (dst_dir / "conversion_meta.json").write_text(
        json.dumps(conversion_meta, indent=2) + "\n",
        encoding="utf-8",
    )

    report["pretrained_model"] = str(dst_dir)
    return report
