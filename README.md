# LeRobot Inference

将旧版 `policy_train` 的 ACT 产物转换为 LeRobot checkpoint，并以 CLI 或 Dora 节点运行在线推理；同时支持 FastWAM 在 LIBERO 中进行闭环评估。

```text
policy_train ACT ── convert ──> pretrained_model/ ── infer ──> Dora JointCommand
LeRobot checkpoint ──────────────────────────────────┘
```

推理始终加载完整的 `pretrained_model/` 目录（权重、配置和 processor），而不是单独的权重或统计文件。

## 快速开始

要求：Linux x86_64、Python 3.12、[uv](https://docs.astral.sh/uv/)；GPU 推理需要兼容 lock 中稳定版 CUDA 13 Torch wheel 的 NVIDIA 驱动。Torch、TorchVision 和 CUDA 用户态依赖由 `uv.lock` 唯一固定，不再使用 nightly/cu128 专用索引或手写 NVIDIA wheel 列表。安装时需能访问内部 Forge GitLab 仓库。

当前 lock 固定 `torch 2.11.0+cu130`、`torchvision 0.26.0+cu130`、`triton 3.6.0` 和 `numpy 2.2.6`，与 `lerobot_trainer` 的核心版本对齐。RTX 5060（compute capability 12.0）已通过 CUDA matrix multiplication、源码 ACT `infer-once` 和 PyInstaller ACT `infer-once`；Torch wheel 包含 `sm_75/sm_80/sm_86/sm_90/sm_100/sm_120`，但 RTX 30/40 系列仍需分别完成真实模型验收后才能标记为 tested。

```bash
git clone <repository-url>
cd lerobot_runner
uv sync --extra dev

uv run lerobot --help
uv run pytest
```

主命令为 `lerobot`，子命令：

```bash
uv run lerobot convert --help
uv run lerobot infer --help
uv run lerobot infer-once --help
uv run lerobot eval-libero --help
```

兼容命令 `lerobot-convert`、`lerobot-infer` 和 `lerobot-infer-once` 仍可使用。

## 项目结构

```text
.
├── cli.py                 # 统一 CLI 入口
├── common/                # checkpoint 与预训练资产共用逻辑
├── convert/               # policy_train ACT → LeRobot checkpoint
├── inference/             # Dora、单次推理与 LIBERO 闭环评估实现
│   └── policies/          # ACT、Pi0.5、FastWAM 策略适配器
├── config/                # 按策略拆分的转换与推理配置模板
├── examples/              # 转换、7D 与 14D Dora 示例
├── scripts/               # 安装、测试、打包辅助脚本
├── tests/                 # 单元测试
└── docs/                  # 设计与实现说明
```

## 转换 ACT 权重

源目录至少应包含：

```text
<src-dir>/
├── policy_epoch_*_standard.safetensors  # 新版，或单个旧版 policy.ckpt
└── dataset_stats.pkl
```

支持 tensor-only `.safetensors` 和旧版 `.ckpt`；legacy `.ckpt` 使用 `torch.load(weights_only=True)`，不会执行 checkpoint 中的任意 Python 对象。`task.json` 用于相机名称和 ACT 超参数，推荐提供训练时使用的文件；省略时使用默认的 left/right/top 三相机与 ACT 参数。

```bash
uv run lerobot convert --config examples/dora_convert/convert.yaml
```

YAML 中的相对路径相对于 YAML 文件所在目录。转换输出目录可直接用作 `policy.pretrained_path`：

```text
<dst-dir>/
├── model.safetensors
├── config.json
├── policy_preprocessor.json
├── policy_postprocessor.json
└── conversion_meta.json
```

## 运行推理

每个已实现的策略都有独立模板：

| 配置 | backend |
| --- | --- |
| `config/inference/act.yaml` | ACT 标准 chunk，可选 temporal ensemble |
| `config/inference/pi05.yaml` | PI0.5 同步 `select_action()` |
| `config/inference/pi05_async_rtc.yaml` | PI0.5 异步 Real-Time Chunking |
| `config/inference/fastwam_libero.yaml` | FastWAM 同步 `select_action()`、严格离线加载 |
| `config/inference/fastwam_robotwin.yaml` | FastWAM RoboTwin 组合图像、14D 单次推理 |

完整说明见 `config/inference/README.md`。

推理配置必须确保：

1. `joints` 的顺序和数量与模型 action 维度一致（7D 对应 7 个关节，14D 对应 14 个关节）。
2. 默认使用 `joints` 构造 `observation.state`；若 state 与 action 的维度或顺序不同，单独配置顶层 `state_joints`。
3. `image_inputs` 的 key 可以是任意 Dora image topic/input ID；alias 用于匹配 checkpoint 的 `observation.images.<alias>`。ACT 和 PI0.5 都使用运行时与 checkpoint 相机的交集（至少需一路匹配），额外 topic 会被忽略。减少训练时使用的视角可能降低策略效果；PI0.5 会为缺失视角生成 masked empty image。
4. `policy.pretrained_path` 指向包含 `model.safetensors` 的目录。
5. `policy.torch_compile` 默认关闭；启用时应在控制开始前完成首次 forward warmup。

```bash
# 不启动 Dora 的单次推理验证
uv run lerobot infer-once --config examples/dora_infer_act_14d/policy_act.yaml

# 真机 Dora 示例：先按设备和模型修改 YAML
cd examples/dora_infer_act_14d
dora run dataflow.yaml
```

可参考 `examples/dora_infer_act_7d/` 和 `examples/dora_infer_act_14d/` 的完整 dataflow 与策略配置。

Pi0.5 还需要本地 tokenizer 路径和语言指令：

```yaml
policy:
  type: pi05
  pretrained_path: /path/to/pretrained_model
  tokenizer_path: /path/to/local-paligemma-tokenizer
  instruction: pick up the object and place it in the box
```

Pi0.5 提供两个显式 backend：默认 `inference_mode: sync` 使用 LeRobot 原生
`select_action()`；`inference_mode: async_rtc` 使用后台 `predict_action_chunk()`、
`LatencyTracker` 和 `ActionQueue.merge()` 实现非阻塞 Real-Time Chunking。异步实现
不访问 policy 私有 action queue，也不使用旧版按队列长度跳帧的启发式逻辑。
切换 instruction、pause、stop 或 reset 时会丢弃旧 chunk。加载过程保持 pretrained
目录只读，并在权重损坏或与 config 不兼容时直接失败，不会回退到随机初始化模型。
完整配置见 `examples/dora_sim_infer_pi05_7d_bin/`。

### FastWAM 单次推理

当前 FastWAM 接入仅提供同步 `select_action()` 和 `infer-once`，尚未接入 Dora。
除 LeRobot checkpoint 外，还必须准备本地 Wan2.2 Diffusers VAE、UMT5 text encoder
及 tokenizer。加载过程强制 `strict=True`，任何权重、Processor 或离线资产缺失都会
直接失败，不允许联网下载或跨 embodiment 随机初始化。

LIBERO 模板使用 8 维 state、7 维 action，以及 `image`、`image2` 两路 224×224
RGB 图像。checkpoint 不包含关节名称，因此模板中的 state/action 名称仅供冒烟；
接入 PaOS 真机前必须按训练数据确认语义、单位和顺序。

```bash
HF_HUB_OFFLINE=1 uv run lerobot infer-once \
  --config config/inference/fastwam_libero.yaml \
  --height 224 \
  --width 224
```

模板中的 `policy.wan_diffusers_path` 指向包含 `vae/`、`text_encoder/` 的本地
snapshot，`policy.tokenizer_path` 指向本地 `google/umt5-xxl` 目录。首次真实推理
需要可用的 NVIDIA GPU；应记录模型加载时间、首次推理时间及 CPU/GPU 内存峰值。

RoboTwin 模板使用 14 维 state/action。模型名中的 `3cam` 表示训练时的三相机
组合输入，但 checkpoint 对外只定义一个 `(3,384,320)` 的
`observation.images.image`；运行时必须先按训练管线生成该组合图像：

```bash
HF_HUB_OFFLINE=1 uv run lerobot infer-once \
  --config config/inference/fastwam_robotwin.yaml \
  --height 384 \
  --width 320
```

### FastWAM × RoboTwin 闭环评估

RoboTwin 闭环评估采用“RoboTwin 环境适配层 + `lerobot_runner` 推理后端”的分层方式：

```text
RoboTwin/SAPIEN（物理 GPU 1）
    │ 三相机 RGB、14D joint state、动态 instruction
    ▼
RoboTwin/XPolicyLab/policy/FastWAM_LeRobot
    │ WebSocket
    ▼
lerobot_runner/.venv/bin/python（物理 GPU 0）
    │ FastWAMPolicyAdapter.from_pretrained()
    ▼
LeRobot checkpoint + Wan2.2 + UMT5
    │ 10×14 joint action chunk
    └──────────────────────────────> RoboTwin rollout
```

`lerobot_runner` 在此流程中提供 Python/uv 运行环境、严格离线资产加载、LeRobot
processor、FastWAM 推理和原生 action queue；三相机拼图、RoboTwin state/action
封装、仿真、视频和结果汇总位于 RoboTwin 的独立 XPolicyLab 适配器中。现有
`RoboTwin/XPolicyLab/policy/FastWAM` 面向原始 `.pt` checkpoint，不参与该流程。

当前接入固定为：

- `place_a2b_left`
- `demo_clean` 与 seen instruction
- `aloha_agilex`、`joint`、双臂 `6+1+6+1`
- 每次规划 10 步，state/action 均为 14 维
- 物理 GPU 0 运行策略，物理 GPU 1 运行 SAPIEN
- 不支持 batch、CPU 或单 GPU 回退

假定 `RoboTwin` 与 `lerobot_runner` 已准备在本机目录中。先安装策略 WebSocket
运行依赖：

```bash
cd /path/to/lerobot_runner
uv sync --frozen --extra dev --extra robotwin
```

如果当前环境还需要保留 LIBERO 评估依赖，可同时传入
`--extra libero-eval`。随后配置本地路径；所有模型资源必须已经完整下载：

```bash
export RUNNER_ROOT=/path/to/lerobot_runner
export ROBOTWIN_ROOT=/path/to/RoboTwin
export FASTWAM_CHECKPOINT=/path/to/fastwam_robotwin_checkpoint
export FASTWAM_WAN_DIFFUSERS_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers/snapshot
export FASTWAM_TOKENIZER_PATH=/path/to/umt5-xxl

cd "${ROBOTWIN_ROOT}/XPolicyLab/policy/FastWAM_LeRobot"
```

先通过真实 FastWAM 完成一次 XPolicyLab WebSocket 协议检查。它会验证动态指令、
三相机输入以及有限的 `10×14` action chunk：

```bash
bash protocol_smoke.sh "${FASTWAM_CHECKPOINT}" "${RUNNER_ROOT}" 0
```

协议检查通过后，使用环境 seed 参数 `0` 执行单 episode 冒烟。RoboTwin 实际从
seed `100000` 开始：

```bash
FASTWAM_EVAL_TEST_NUM=1 bash eval.sh \
  RoboTwin place_a2b_left "${FASTWAM_CHECKPOINT}" \
  aloha_agilex joint 0 0 1 "${RUNNER_ROOT}" RoboTwin
```

最后使用独立环境 seed 参数 `1` 执行 5 个正式 episode，实际从 seed `200000`
开始，避免重复冒烟场景：

```bash
FASTWAM_EVAL_TEST_NUM=5 bash eval.sh \
  RoboTwin place_a2b_left "${FASTWAM_CHECKPOINT}" \
  aloha_agilex joint 1 0 1 "${RUNNER_ROOT}" RoboTwin
```

`eval.sh` 会在模型加载前检查双 GPU、CUDA/BF16、离线资产、SAPIEN 渲染、FFmpeg
H.264 和磁盘空间；任一检查失败都会停止，不会降级到 CPU。策略请求超时固定为
300 秒，评估视频沿用 RoboTwin 原生头部相机 10 FPS H.264 输出。

成功完成后，RoboTwin 创建的原生结果目录包含：

```text
<robotwin-result-dir>/
├── _result.txt
├── episode*.mp4
├── summary.json
├── artifact_manifest.json
├── actions/
│   └── episode_<seed>.jsonl
├── policy_server.log
└── client.log
```

`summary.json` 包含完成数、成功数、成功率、各 episode 结果、规划次数以及平均、
P50、P95 推理时延。成功率没有最低阈值；即使为 0，只要所有 rollout 完整、动作
有限且产物齐全，技术接入仍视为通过。发生 WebSocket 超时、OOM、rollout 异常或
NaN/Inf 时会保留 staging 日志和已有 trace，但不会生成技术通过的汇总文件。

### FastWAM × LIBERO 闭环评估

`eval-libero` 使用当前 `FastWAMPolicyAdapter` 在真实 LIBERO MuJoCo 环境中执行
固定初始状态闭环评估，并输出成功状态、逐步动作、运行日志和双相机视频。LIBERO
依赖位于独立的 `libero-eval` extra 中，不需要安装 LIBERO 仓库自带的旧版
`requirements.txt`。

先同步评估依赖，并设置本地 LIBERO 仓库路径：

```bash
uv sync --extra dev --extra libero-eval

export LIBERO_ROOT=/path/to/LIBERO
```

运行默认场景 `libero_spatial / task_id=0 / init_state_id=0`：

```bash
MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --extra libero-eval lerobot eval-libero \
  --config examples/fastwam_libero_eval/policy_fastwam.yaml \
  --libero-root "${LIBERO_ROOT}" \
  --suite libero_spatial \
  --task-id 0 \
  --episodes 1
```

默认参数为两路 `256×256` 相机、10 个稳定步、20 Hz 控制频率、每次执行 10 步
action chunk，以及最多 300 个动作步。运行结果写入：

```text
examples/fastwam_libero_eval/out/<timestamp>/
├── result.json
├── actions.jsonl
├── rollout.mp4
├── run.log
└── runtime_config.yaml
```

任务未成功但正常运行到 horizon 时命令返回 0；模型加载、环境创建、非有限 action
或视频写入失败时返回非零。详细参数和输出字段见
`examples/fastwam_libero_eval/README.md`。

## 打包

```bash
bash scripts/build.sh
```

PyInstaller 产物位于 `dist/lerobot_infer/lerobot_infer`：

```bash
dist/lerobot_infer/lerobot_infer infer --config /path/to/policy_act.yaml
```

构建缓存和产物均被 Git 忽略；需要重新打包时可执行 `bash scripts/build.sh --clean`。
