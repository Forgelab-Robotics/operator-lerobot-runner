# RoboCasa + SmolVLA 12D（Dora 推理示例）

官方 `lerobot/smolvla_robocasa` 策略的 Dora 推理示例（SmolVLA = SmolVLM2-500M
视觉语言骨干 + flow-matching 动作专家，指令条件；每次推理产出 50 步动作 chunk，
由 LeRobot 的队列逐步吐出）。

```text
robocasa_benchmark → proprio_state / image×3 / policy_command → policy
                   ←──────────────── action ──────────────────┘
```

## 1. 模型来源与许可证（完成标准记录）

| 组件 | 来源 | 仓库 | 许可证 |
|------|------|------|--------|
| Policy checkpoint | [lerobot/smolvla_robocasa](https://huggingface.co/lerobot/smolvla_robocasa) | official LeRobot 格式（config.json + model.safetensors + pre/postprocessor），866 MB | 见仓库 |
| VLM 骨干 | [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | 由 checkpoint `vlm_model_name` 引用，离线部署时本地化 | Apache-2.0 |
| 仿真环境 | [robocasa/robocasa](https://github.com/robocasa/robocasa) + [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) | robosuite **1.5.2**（注意与 LIBERO 的 1.4.0 冲突） | MIT |

**固定 revision 要求**：部署前记录各模型的实际 revision，权重下载到本地目录，
避免部署时依赖网络。

## 2. 绑定模型（本地路径）

```bash
cd examples/dora_sim_infer_smolvla_robocasa_12d
ln -s /path/to/smolvla_robocasa model
```

`model`、`logs/`、`out/` 均被 Git 忽略，不会提交机器相关路径。

模型必须匹配：

- state：**16D**；action：**12D**
- image features：`observation.images.camera1/2/3`
- LeRobot processor schema（含归一化统计文件）

## 3. 观测与动作口径（三个必须核对的点）

**① `config.json` 声明的 state 维度是错的。**
官方 checkpoint 里 `observation.state` 写的是 `[6]`，但模型实际要 **16 维**。
实测喂 6 维报 `size of tensor a (6) must match b (16)`。本适配器**以归一化
统计量为准**，加载时会给出警告：

```text
WARNING: SmolVLA checkpoint declares observation.state=6 but its normalization
statistics are 16-dimensional; trusting the statistics. Feed 16-D state.
```

16 维的组成（与官方 lerobot wrapper 逐位一致）：
`base_position(3) + base_rotation(4) + eef_position_relative(3) + eef_rotation_relative(4) + gripper_qpos(2)`

判据来自归一化统计量本身：第 3 维均值 0.701 且范围仅 0.6992–0.7586（底盘高度恒定），
第 4/5 维恒为 0（只绕 z 轴转的底盘，四元数 x/y 必为 0），第 15/16 维 ±0.029 完美对称（两指夹爪）。

**② 动作 12 维的分量语义。**
底盘 4 + **控制模式 1** + 末端位姿 6 + 夹爪 1。其中「控制模式」是**离散切换量**
（切底盘/切手臂），不是连续控制；官方训练数据里它的均值是 −0.780（绝大多数时间
固定在一侧），第 4 维恒为 0。全部分量落在 `[-1, 1]`，与环境的 `Box(-1,1,12)` 一致。

**③ 相机映射与官方 rename_map 对齐。**
`camera1`=`robot0_agentview_left`、`camera2`=`robot0_eye_in_hand`、
`camera3`=`robot0_agentview_right`。

## 4. 冒烟测试（单次推理，可独立跑通）

```bash
# 源码环境（conda lerobot + PYTHONPATH 包布局）
uv run lerobot infer-once --config ./policy_smolvla.yaml

# 打包二进制（bash scripts/build.sh 后）
../../bin/lerobot_infer/lerobot_infer infer-once --config ./policy_smolvla.yaml
```

## 5. adapter 可用性演示（无需仿真，可独立复现）

本节命令**不依赖任何仿真/benchmark**：加载 SmolVLA adapter、用形状匹配的假观测
跑一步推理，输出动作 JSON。用于证明 adapter 可用（能加载、能推理、输出形状正确）。

```bash
cd <LEROBOT_RUNNER>
export PYTHONPATH=<PKG_LINK>
export HF_HUB_OFFLINE=1
python <PKG_LINK>/lerobot_inference/inference/run_once.py \
  --config examples/dora_sim_infer_smolvla_robocasa_12d/policy_smolvla.yaml \
  --height 256 --width 256
```

实测输出（2026-08-18，RTX 6000D）：

```text
Loading  HuggingFaceTB/SmolVLM2-500M-Video-Instruct weights ...
WARNING:...smolvla.adapter:SmolVLA checkpoint declares observation.state=6 but its
  normalization statistics are 16-dimensional; trusting the statistics. Feed 16-D state.
Loading weights from local directory
{
  "policy_type": "smolvla",
  "pretrained_path": ".../smolvla_robocasa",
  "action": [
    0.005291203036904335, -0.0005800274666398764, -0.0009515160345472395, 0.0,
    -0.9994803071022034, 0.8429775238037109, 0.07071162015199661, -0.7390366196632385,
    -0.10553150624036789, 0.0005430141463875771, -0.25530341267585754, -0.9544166326522827
  ],
  "action_dim": 12
}
```

**为什么这能证明 adapter 可用**：

1. **官方入口**：`run_once.py` 内部走 `load_config` → `create_policy_adapter`
   （与 `lerobot infer` Dora 节点同一加载路径，非独立脚本）。
2. **真实 checkpoint**：`pretrained_path` 指向本地 `smolvla_robocasa`，权重全部
   加载，无随机初始化回退（`Loading weights from local directory`）。
3. **形状与分布均正确**：`action_dim: 12` 与 RoboCasa 动作维度一致；第 4 维为
   `0.0`、第 5 维（控制模式）为 `-0.999`，两者都与官方训练数据的分布吻合
   （前者恒为 0，后者均值 −0.780）。
4. **离线**：`HF_HUB_OFFLINE=1`，全程无网络访问。

## 6. 资源占用（实测，RTX 6000D 85GB）

| 指标 | 数值 |
|------|------|
| 模型加载耗时 | 6.5 s |
| 首次推理（生成 50 步 chunk 第 1 步） | 923 ms |
| chunk 内后续步（队列出队） | ~1 ms |
| GPU 显存峰值（allocated / reserved） | 0.91 / 0.93 GB |
| CPU 常驻内存峰值（VmHWM） | 3271 MB |
| `n_action_steps` | 50 |

显存占用比 VLA-JEPA（6.18 GB）低一个量级，因为 SmolVLM2 只有 500M 参数。
`n_action_steps=50` 意味着每 50 步才推理一次，稳态平均延迟很低。

## 7. 与 benchmark 节点联跑

benchmark 侧用 `forge_runtime` 的 RoboCasa 节点，**两个节点必须用不同的 conda 环境**：
RoboCasa 要 robosuite 1.5.2，而 LIBERO 要 1.4.0，装在一起会互相破坏。
dora 支持每个节点指定各自的解释器，见 `dataflow.yaml`。

```bash
export PATH=<FORGE_BENCH_ENV>/bin:$PATH   # dora CLI
dora run dataflow.yaml
```

## 8. 已知结果（2026-08-18 实测）

在 `CloseFridge` 任务上跑 10 个 episode：

| 评测方式 | success_rate | 耗时 |
|:---|---:|---:|
| 本节点（Forge 两节点） | **0/10 = 0.0%** | 445 s |
| 官方 `lerobot-eval` | **`pc_success: 0.0`** | 442 s |

**两者结果一致**，官方日志里 `avg_max_reward` 也是 0.0（连部分进展都没有）。
这说明接入无误——同一份权重、同一个任务，官方评测代码得到相同结论。

**这个 checkpoint 在该任务上确实是 0%**，并非接入问题。注意 lerobot 官方 CI
只跑「10 个 atomic 任务各 1 个 episode」的冒烟，从未声称该 checkpoint 的成功率；
RoboCasa atomic 任务约 49% 的 SOTA 是其他方法（PointMapPolicy）的成绩。

排查过程中已逐条排除：state 维度、图像朝向（与官方 `process_img` 逐位相同）、
`max_steps`（完整 horizon 1050 步同样结果）、动作区间、控制模式分量、指令下发、
相机映射。录像的帧间差分显示机器人全程在动（运动量 4.5–7.8，静止帧占比 0.0–0.3%），
属于「在动但完不成任务」，而非卡死或链路中断。
