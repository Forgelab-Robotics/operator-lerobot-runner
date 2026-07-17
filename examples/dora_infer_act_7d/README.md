# dora_infer_act_7d

真机 **单臂 ACT（7D）** 示例，格式对齐 [`single_real_inference`](../../single_real_inference/)，策略节点为本仓库 `lerobot-infer`。

## 文件

| 文件 | 说明 |
|------|------|
| `dataflow.yaml` | 相机 / Piper / task_robot / policy / 录制 |
| `policy_act.yaml` | 本项目推理配置（`pretrained_path`） |
| `task_robot.yaml` | 右臂 7 关节，与 policy joints 一致 |
| `piper_slave_*.yaml` | CAN 口与初位姿 |
| `camera_*.yaml` | `/dev/video*` 设备 |
| `convert/task.json` | 转换用示例 task（三路 left/right/top + ACT 超参） |
| `convert/convert.yaml` | 转换配置模板（改 `src_dir` / `dst_dir`） |

## 前置

1. `bash scripts/setup.sh`（本仓库）
2. 同级目录存在可用的 `forge_runtime`（`dataflow.yaml` 默认用 `../../../forge_runtime/bin/...` 二进制）
3. 本仓库执行 `bash scripts/build.sh`，生成 `bin/lerobot_infer/lerobot_infer`
4. 将 `policy_act.yaml` 的 `pretrained_path` 指到 **7D** `pretrained_model/`

### 可选：从 policy_train 转换

```bash
# 先改 convert/convert.yaml 里的 src_dir / dst_dir
uv run lerobot convert --config examples/dora_infer_act_7d/convert/convert.yaml
# 或：
# bin/lerobot_infer/lerobot_infer convert --config examples/dora_infer_act_7d/convert/convert.yaml
```

`convert/task.json` 相机别名与 `policy_act.yaml` 的 `image_inputs`（left / right / top）一致；有真实训练 `task.json` 时优先用那份。

## 运行

```bash
cd /path/to/lerobot_inference

# 可选：无真机 smoke
uv run lerobot infer-once --config examples/dora_infer_act_7d/policy_act.yaml

# 真机 Dora
cd examples/dora_infer_act_7d
dora run dataflow.yaml
```

## 约定

- `len(joints) == 7`，与模型 state/action 维数一致
- 仅右臂接 `action`；左臂节点保留但默认不控
- `image_inputs` 别名：`left` / `top` / `right`，须与训练相机 key 对应
