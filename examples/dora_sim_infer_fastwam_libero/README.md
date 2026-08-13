# FastWAM LIBERO Forge/Dora 协议仿真

该示例验证 FastWAM LIBERO checkpoint 的 Forge 消息与 Dora 生命周期集成：

```text
protocol simulator ── JointState + image/image2 ──> lerobot policy
                   <─────── JointCommand ──────────┘
                   ─────── PolicyCommand ─────────>
                   <── PolicyCommandStatus ────────
```

协议模拟器不是 LIBERO 物理环境，不用于评估任务成功率。它固定发布 8D state、
两路 224×224 RGB 图像，接收 7D action，并将前 7 个 action 映射回前 7 个 state。

## 1. 准备环境和本地资产

```bash
cd /path/to/lerobot_runner
uv sync --extra dev
cd examples/dora_sim_infer_fastwam_libero

ln -s ../../checkpoints/fastwam_libero_uncond_2cam224 model
ln -s ../../checkpoints/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635 wan
ln -s ../../checkpoints/fastwam_runtime/google/umt5-xxl tokenizer
export HF_HUB_OFFLINE=1
```

`instruction` 必须替换为待验证 LIBERO task 在训练数据中的原始文本。

## 2. 分阶段验证

### simulator 与 observation 路由

```bash
dora run dataflow_simulator.yaml
```

日志出现以下内容表示 state 和两路图像均可被 Forge 消息解码：

```text
OBSERVATION_READY state_dim=8 images=['image/camera_0', 'image/camera_1']
```

### Policy 单次推理

```bash
../../.venv/bin/lerobot infer-once \
  --config ./policy_fastwam.yaml --height 224 --width 224
```

### 手动生命周期和完整闭环

直接启动手动生命周期 dataflow：

```bash
HF_HUB_OFFLINE=1 dora run dataflow.yaml
```

启动后 Policy 保持 idle，不应产生 action。在另一终端依次触发：

```bash
touch control/start
cat control/status.json
cat control/last_action.json

touch control/pause
touch control/resume
touch control/reset
touch control/start
touch control/stop
```

`status.json` 应显示命令为 `done`；收到首个有限 7D JointCommand 后，模拟器日志会
打印 `complete JointCommand loop received`。

### 最后启用自动启动

只有上述阶段全部通过后才运行：

```bash
HF_HUB_OFFLINE=1 dora run dataflow_auto.yaml
```

## 文件说明

- `dataflow_simulator.yaml`：模拟器与 observation probe。
- `dataflow.yaml` / `policy_fastwam.yaml`：手动生命周期，`auto_start: false`。
- `dataflow_auto.yaml` / `policy_fastwam_auto.yaml`：最终自动启动。
- `simulator.yaml`：8D/7D 映射及双图像协议。
