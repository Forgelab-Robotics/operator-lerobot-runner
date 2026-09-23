# FastWAM RoboTwin Forge/Dora 协议仿真

该示例验证 FastWAM RoboTwin checkpoint 的 Forge 消息与 Dora 生命周期集成：

```text
protocol simulator ── 14D JointState + combined image ──> lerobot policy
                   <────────── 14D JointCommand ──────────┘
                   ─────────── PolicyCommand ────────────>
                   <────── PolicyCommandStatus ───────────
```

协议模拟器不是 RoboTwin 物理环境，不用于评估任务成功率。checkpoint 名称包含
`3cam`，但对外只有一个 `(3,384,320)` 的 `observation.images.image`；本示例发布
一张确定性的组合图像，不猜测真实训练管线的三相机拼接方式。

## 1. 准备环境和本地资产

```bash
cd /path/to/lerobot_runner
uv sync --extra dev
cd examples/dora_sim_infer_fastwam_robotwin

ln -s ../../checkpoints/fastwam_robotwin_uncond_3cam_384 model
ln -s ../../checkpoints/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/b8fff7315c768468a5333511427288870b2e9635 wan
ln -s ../../checkpoints/fastwam_runtime/google/umt5-xxl tokenizer
export HF_HUB_OFFLINE=1
```

`instruction` 必须替换为待验证 RoboTwin task 在训练数据中的原始文本。

## 2. 分阶段验证

### simulator 与 observation 路由

```bash
dora run dataflow_simulator.yaml
```

日志出现以下内容表示 14D state 和组合图像均可被解码：

```text
OBSERVATION_READY state_dim=14 images=['image/combined']
```

### Policy 单次推理

```bash
../../.venv/bin/lerobot infer-once \
  --config ./policy_fastwam.yaml --height 384 --width 320
```

### 手动生命周期和完整闭环

```bash
HF_HUB_OFFLINE=1 dora run dataflow.yaml
```

启动后 Policy 保持 idle，不应产生 action。在另一终端依次执行：

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

`status.json` 应显示命令为 `done`；收到首个有限 14D JointCommand 后，模拟器日志
会打印 `complete JointCommand loop received`。

### 最后启用自动启动

```bash
HF_HUB_OFFLINE=1 dora run dataflow_auto.yaml
```

## 文件说明

- `dataflow_simulator.yaml`：模拟器与 observation probe。
- `dataflow.yaml` / `policy_fastwam.yaml`：手动生命周期，`auto_start: false`。
- `dataflow_auto.yaml` / `policy_fastwam_auto.yaml`：最终自动启动。
- `simulator.yaml`：14D 一一映射及组合图像协议。
