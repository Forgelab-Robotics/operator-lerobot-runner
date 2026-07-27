# Piper MuJoCo + ACT 7D（二进制运行）

这个示例使用 `lerobot_runner/bin/` 中的本地二进制完成端到端仿真推理：

```text
mujoco_sim → task_robot → lerobot_infer → task_robot → mujoco_sim
     └──────────────── image_viewer ────────────────┘
```

场景为 `bin/050f9b6f-30e7-4dab-a4c4-0a4aed9b8adb_scene/scene.xml`，策略输入为 7D state 和 `left/right/top` 三路 640×480 RGB 图像。

## 1. 准备策略二进制

仓库已有 `mujoco_sim`、`task_robot`、`image_viewer`。首次运行还需在 `lerobot_runner` 根目录构建策略二进制：

```bash
bash scripts/build.sh
```

产物会复制到：

```text
bin/lerobot_infer/lerobot_infer
```

## 2. 绑定模型

`policy_act.yaml` 使用本目录下的 `model`。可将匹配 7D state/action 和 `left/right/top` 三路 image feature 的 LeRobot ACT 产物链接到这里：

```bash
cd examples/dora_sim_infer_act_7d_bin
ln -s /path/to/pretrained_model model
```

`model` 已加入 `.gitignore`，不会提交机器相关路径。

## 3. 冒烟测试

先确认策略二进制和模型能够独立推理：

```bash
../../bin/lerobot_infer/lerobot_infer infer-once --config ./policy_act.yaml
```

## 4. 运行仿真

```bash
cd examples/dora_sim_infer_act_7d_bin
dora run dataflow.yaml
# 若 dora 未加入 PATH，可从当前目录运行：
# ../../../../../forge_runtime/.venv/bin/dora run dataflow.yaml
```

策略配置为 `auto_start: true`，不依赖 gateway；启动后会在收到完整 observation 时自动推理并驱动 MuJoCo。按 `Ctrl+C` 停止。

## 配置对应关系

- `simulator.yaml`：MuJoCo joint、gripper 派生状态及三路相机映射。
- `task_robot.yaml`：转发 simulation state/image，并把 policy `JointCommand` 下发给 simulator。
- `policy_act.yaml`：7D ACT、模型路径和 checkpoint image feature alias。
  `temporal_ensemble_coeff: null` 表示使用模型的标准 action chunk；设置为
  `0.01` 会启用 LeRobot temporal ensemble，并自动将有效 `n_action_steps` 设为 `1`。
- `dataflow.yaml`：所有节点均引用 `../../bin/`，不依赖 `forge_runtime/bin`。

如果替换模型，必须确认 state/action 顺序均为：

```text
joint1, joint2, joint3, joint4, joint5, joint6, gripper
```
