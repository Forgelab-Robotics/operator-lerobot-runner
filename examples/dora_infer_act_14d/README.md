# dora_infer_act_14d

真机 **双臂 ACT（14D）** 示例，格式对齐 [`single_real_inference`](../../single_real_inference/)，策略节点为本仓库 `lerobot-infer`。

## 文件

| 文件 | 说明 |
|------|------|
| `dataflow.yaml` | 相机 / 左右 Piper / task_robot / policy / 录制 |
| `policy_act.yaml` | 本项目推理配置（`pretrained_path`） |
| `task_robot.yaml` | 左+右共 14 关节，与 policy joints 一致 |
| `piper_slave_*.yaml` | CAN 口与初位姿 |
| `camera_*.yaml` | `/dev/video*` 设备 |

## 前置

1. `bash scripts/setup.sh`（本仓库）
2. 同级目录存在可用的 `forge_runtime`（`dataflow.yaml` 默认用 `../../../forge_runtime/bin/...` 二进制）
3. 本仓库执行 `bash scripts/build.sh`，生成 `bin/lerobot_infer/lerobot_infer`
4. 将 `policy_act.yaml` 的 `pretrained_path` 指到 **14D** `pretrained_model/`  
   （转换产物默认：`test_convert/out_put/converted`）

## 运行

```bash
cd /path/to/lerobot_inference

# 可选：无真机 smoke
uv run lerobot infer-once --config examples/dora_infer_act_14d/policy_act.yaml

# 真机 Dora
cd examples/dora_infer_act_14d
dora run dataflow.yaml
```

## 约定

- `len(joints) == 14`（左 7 + 右 7），顺序与训练一致
- 左右臂均接 `task_robot` → Piper `action`
- `image_inputs` 别名：`left` / `top` / `right`
