"""Policy 节点配置加载与校验（对标 pick_and_place/config.py）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lerobot_inference.common.paths import resolve_user_path


@dataclass(frozen=True)
class JointConfig:
    name: str
    mode: str = "position"
    unit: str = "radians"
    type: str = "actuator"


@dataclass
class PolicyNodeConfig:
    """Policy 节点完整配置：joints、policy、image_inputs（input_id -> alias）。"""

    joints: list[JointConfig]
    policy: dict[str, Any]
    auto_start: bool = False
    image_input_id_to_alias: dict[str, str] = field(default_factory=dict)

    @property
    def joint_order(self) -> list[str]:
        """关节名称顺序（与 task_robot / simulator 一致）。"""
        return [j.name for j in self.joints]

    @property
    def image_input_ids(self) -> set[str]:
        return set(self.image_input_id_to_alias.keys())

    @property
    def alias_for_cameras(self) -> list[str]:
        """相机别名顺序；由 YAML image_inputs 定义顺序决定。"""
        return list(self.image_input_id_to_alias.values())

    def runtime_policy_config(self) -> dict[str, Any]:
        """生成策略运行时配置，补齐可由节点配置推断的字段。"""
        policy_config = {
            **self.policy,
            "auto_start": self.auto_start,
            "camera_names": self.alias_for_cameras,
            "joint_count": len(self.joints),
        }
        # 与 pick_and_place 相同：ACT 的 state_dim 必须等于 joints 数量
        ptype = str(policy_config.get("type", "")).strip()
        if ptype in {"ACT", "act"}:
            inferred_state_dim = len(self.joints)
            raw_state_dim = policy_config.get("state_dim", inferred_state_dim)
            try:
                state_dim = int(raw_state_dim)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"policy.state_dim 必须为整数: {raw_state_dim!r}") from exc
            if state_dim != inferred_state_dim:
                raise ValueError(
                    f"policy.state_dim={state_dim} 与 joints 数量={inferred_state_dim} 不一致"
                )
            policy_config["state_dim"] = state_dim
        return policy_config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyNodeConfig:
        joints_raw = data.get("joints")
        if not joints_raw:
            raise ValueError("joints 或 joint_order 不能为空")

        joints: list[JointConfig] = []
        for item in joints_raw:
            if isinstance(item, str):
                joints.append(JointConfig(name=item))
            elif isinstance(item, dict) and "name" in item:
                joints.append(
                    JointConfig(
                        name=str(item["name"]),
                        mode=str(item.get("mode", "position")),
                        unit=str(item.get("unit", "radians")),
                        type=str(item.get("type", item.get("joint_type", "actuator"))),
                    )
                )
            else:
                raise ValueError(f"joint 项须为字符串或包含 name 的字典: {item!r}")

        policy = data.get("policy", {})
        if not policy or not policy.get("type"):
            raise ValueError("policy 必须包含 type（ACT / pi05 / smolvla 等）")

        image_inputs = data.get("image_inputs")
        if not isinstance(image_inputs, dict) or not image_inputs:
            raise ValueError("image_inputs 不能为空，且必须为 dict[str, str]")

        image_input_id_to_alias = {
            str(input_id): str(alias) for input_id, alias in image_inputs.items()
        }

        return cls(
            joints=joints,
            policy=policy,
            auto_start=_as_bool(policy.get("auto_start", data.get("auto_start", False))),
            image_input_id_to_alias=image_input_id_to_alias,
        )

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> PolicyNodeConfig:
        p = resolve_user_path(path)
        if not p.is_file():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not data:
            raise ValueError(f"配置文件为空: {p}")
        # LeRobot 产物路径：相对本 yaml 解析
        base = p.parent
        policy = data.get("policy")
        if isinstance(policy, dict):
            for key in ("run_dir", "pretrained_path", "tokenizer_path", "ckpt_path"):
                value = policy.get(key)
                if not value:
                    continue
                path_value = Path(str(value)).expanduser()
                if not path_value.is_absolute():
                    policy[key] = str((base / path_value).resolve())
        return cls.from_dict(data)


# 兼容旧导入名
InferenceNodeConfig = PolicyNodeConfig


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_config(
    config_path: str | Path | None = None,
    dataflow_config: dict[str, Any] | None = None,
) -> PolicyNodeConfig:
    """
    加载配置（优先级同 pick_and_place）：
    1. dataflow 内联 config / config_path
    2. config_path 参数
    3. 环境变量 LEROOT_INFERENCE_CONFIG
    """
    data: dict[str, Any] | None = None

    if dataflow_config:
        if "config" in dataflow_config:
            data = dataflow_config["config"]
        elif "config_path" in dataflow_config:
            return PolicyNodeConfig.from_yaml_path(dataflow_config["config_path"])

    path = config_path
    if path is None and data is None:
        path = os.environ.get("LEROOT_INFERENCE_CONFIG")

    if path:
        return PolicyNodeConfig.from_yaml_path(path)

    if data:
        return PolicyNodeConfig.from_dict(data)

    raise ValueError(
        "未找到配置。请设置 LEROOT_INFERENCE_CONFIG、"
        "或通过 --config 指定配置文件、或使用 dataflow config。"
    )
