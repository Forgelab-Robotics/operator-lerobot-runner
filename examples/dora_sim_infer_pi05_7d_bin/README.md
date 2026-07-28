# Piper MuJoCo + PI0.5 7D（二进制运行）

这个示例使用 `lerobot_runner/bin/` 中的本地二进制完成端到端仿真推理：

```text
mujoco_sim → task_robot → lerobot_infer → task_robot → mujoco_sim
     └──────────────── image_viewer ────────────────┘
```

场景为 `bin/050f9b6f-30e7-4dab-a4c4-0a4aed9b8adb_scene/scene.xml`。策略输入为 7D state、语言 instruction，以及 `top/angle/left_pillar` 三路 640×480 RGB 图像。

## 1. 准备策略二进制

首次运行或 PI0.5 源码更新后，在 `lerobot_runner` 根目录构建：

```bash
bash scripts/build.sh
```

产物位于：

```text
bin/lerobot_infer/lerobot_infer
```

## 2. 绑定模型和 tokenizer

同步与异步配置都使用本目录下的 `model` 与 `tokenizer` symlink：

```bash
cd examples/dora_sim_infer_pi05_7d_bin
ln -s /path/to/pi05_pretrained_model model
ln -s /path/to/local_paligemma_tokenizer tokenizer
```

模型必须匹配：

- state/action：7D
- image features：`observation.images.top`、`observation.images.angle`、`observation.images.left_pillar`
- 可由 LeRobot 0.6 加载的 processor schema

`model`、`tokenizer` 和 `out/` 均被 Git 忽略，不会提交机器相关路径或运行日志。

## 3. 设置 instruction

`PIPER_SIM_251225` 的旧部署配置使用以下训练任务文本：

```yaml
instruction: Grab the blue cube and then place upon red cube
```

语言条件是模型输入的一部分，不应改写为看似等价的同义句。checkpoint 本身没有保存 task 文本，因此本示例以 `forge_runtime/examples/inference/policy.pi05.example.yaml` 中的已知部署值为准。若替换为其他 checkpoint，必须同步替换为对应训练数据中的 task 文本。

## 4. LeRobot 0.4.4 checkpoint 兼容

`PIPER_SIM_251225` 使用旧 LeRobot PI0.5 实现训练和部署，两个 policy 配置都显式启用了：

```yaml
compatibility_mode: lerobot_0_4_4
```

该模式在 LeRobot 0.6 runtime 中恢复会影响模型输入分布的三项旧语义：

- state prompt 固定补齐到 32 维后再离散化（本模型是 7 个 state + 25 个 padding）；
- language embedding 乘以 `sqrt(hidden_dim)`；
- 图像 tensor 转换及 resize/letterbox 使用 0.4.4 的数值语义。

这些行为只作用于当前 policy 实例，不修改 checkpoint 文件，也不替换 LeRobot 0.6 的 processor pipeline、action queue 或 RTC。使用由 LeRobot 0.6+ 训练的 checkpoint 时应删除此配置，采用默认的 `native` 模式。

## 5. 冒烟测试

先确认模型、tokenizer、processor 和严格权重加载均正常：

```bash
# 同步 select_action
../../bin/lerobot_infer/lerobot_infer infer-once --config ./policy_pi05.yaml

# 异步 RTC；等待后台首个 action，默认超时 120 秒
../../bin/lerobot_infer/lerobot_infer infer-once \
  --config ./policy_pi05_async.yaml --async-timeout 120
```

这个 checkpoint 的权重约 7 GB，实际加载与 forward 需要显著更多显存。本机 8 GB RTX 5060 已验证在模型初始化阶段 CUDA OOM；请在显存足够的环境运行真实推理。

## 6. 运行仿真

```bash
cd examples/dora_sim_infer_pi05_7d_bin

# 同步模式
dora run dataflow.yaml

# LeRobot async RTC 模式
dora run dataflow_async.yaml

# 若 dora 未加入 PATH：
# ../../../../../forge_runtime/.venv/bin/dora run dataflow_async.yaml
```

两套策略均配置为 `auto_start: true`，不依赖 gateway。按 `Ctrl+C` 停止。

## 推理时序

### 同步 `dataflow.yaml`

这是与旧 `pick_and_place` 默认 PI0.5 配置最接近的对照基线：旧配置未启用 `rtc`，使用普通 action chunk。应先确认同步模式能够完成任务，再评估 RTC。

1. 收集一次完整 observation；
2. 通过原生 `select_action()` 同步生成 50-step action chunk；
3. 以 50 Hz 逐步消费 action；
4. queue 耗尽后再生成下一段。

chunk 边界会暂停等待模型推理。

### 异步 RTC `dataflow_async.yaml`

RTC 会对 PI0.5 去噪过程施加 prefix guidance，因此不是把同步推理简单移到后台，动作结果也不保证与同步或旧版非 RTC 路径一致。本示例使用与旧 RTC 实现一致的 `EXP` attention schedule。

1. 每个 tick 将最新 observation 发布给后台线程；
2. 主控制循环通过 LeRobot `ActionQueue.get()` 非阻塞取 action；
3. queue 低于 `queue_threshold` 时，后台调用 `predict_action_chunk()`；
4. 使用 `LatencyTracker` 计算 inference delay；
5. 使用 `ActionQueue.merge()` 和模型 RTC prefix guidance 合并新旧 chunk。

队列尚未就绪时策略返回 `None`，Dora tick 不会阻塞。后台异常会在主线程明确抛出，不会永久等待。

## 配置对应关系

- `simulator.yaml`：Piper joint、gripper 派生状态及三路相机映射。
- `task_robot.yaml`：转发 simulation state/image，并把 policy `JointCommand` 下发给 simulator。
- `policy_pi05.yaml`：同步 7D PI0.5 配置。
- `policy_pi05_async.yaml`：异步 LeRobot RTC 配置。
- `dataflow.yaml` / `dataflow_async.yaml`：同步/异步拓扑；所有节点均引用 `../../bin/`。

state/action 顺序均为：

```text
joint1, joint2, joint3, joint4, joint5, joint6, gripper
```
