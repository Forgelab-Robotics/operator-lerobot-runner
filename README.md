# LeRobot Inference

将旧版 `policy_train` 的 ACT 产物转换为 LeRobot checkpoint，并以 CLI 或 Dora 节点运行在线推理。

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
```

兼容命令 `lerobot-convert`、`lerobot-infer` 和 `lerobot-infer-once` 仍可使用。

## 项目结构

```text
.
├── cli.py                 # 统一 CLI 入口
├── common/                # checkpoint 与预训练资产共用逻辑
├── convert/               # policy_train ACT → LeRobot checkpoint
├── inference/             # Dora 与单次推理实现
│   └── policies/          # ACT、Pi0.5 策略适配器
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

完整说明见 `config/inference/README.md`。

推理配置必须确保：

1. `joints` 的顺序和数量与模型 action 维度一致（7D 对应 7 个关节，14D 对应 14 个关节）。
2. 默认使用 `joints` 构造 `observation.state`；若 state 与 action 的维度或顺序不同，单独配置顶层 `state_joints`。
3. `image_inputs` 的 alias 必须与 checkpoint 的 image feature key 一致。
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

## 打包

```bash
bash scripts/build.sh
```

PyInstaller 产物位于 `dist/lerobot_infer/lerobot_infer`：

```bash
dist/lerobot_infer/lerobot_infer infer --config /path/to/policy_act.yaml
```

构建缓存和产物均被 Git 忽略；需要重新打包时可执行 `bash scripts/build.sh --clean`。

### 薄 Policy Resource（实验）

本仓库始终是可独立开发的 Python 项目。单独 clone 后直接执行：

```bash
uv sync --frozen
uv run lerobot --version

uv sync --frozen --extra dev
uv run pytest
```

共享 Runtime 不是源码开发的前置条件，只用于发布后的 Resource 组合。薄 Policy 与现有 PyInstaller 产物并存，执行以下命令构建：

```bash
bash scripts/build_thin_policy.sh
```

输出是独立的 `lerobot_inference_policy@1.0.2` Resource payload：

```text
dist/lerobot_policy/
├── bin/lerobot
├── bin/check-policy
└── site-packages/
    ├── lerobot_inference/
    └── lerobot_inference-1.0.2.dist-info/
```

构建脚本先构建当前 wheel，再通过 `uv pip install --no-deps --target site-packages` 安装。Policy 的 wheel metadata 和开发锁仍完整声明依赖，但 payload 严格限制为 Runner 自身代码；Torch、LeRobot、NumPy、OpenCV、Forge 和 CUDA 由独立 `forge_lerobot_runtime` Resource 提供。

`bin/lerobot` 不搜索能力包，也不根据父目录猜测其他 Resource。Resource Resolver 必须通过绝对路径 `FORGE_LEROBOT_RUNTIME_ROOT` 显式绑定 Runtime；模型路径由 Policy 配置绑定，TorchVision backbone 使用标准 `TORCH_HOME` 绑定。Python bytecode、Runtime cache 和 Hugging Face cache 写入 `${FORGE_RUN_DIR}/lerobot_policy/cache`，未设置时写入用户 cache。`HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE` 默认设为 `1`，只允许通过明确的 `FORGE_HF_HUB_OFFLINE=0` / `FORGE_TRANSFORMERS_OFFLINE=0` 放开联网。

组合后执行合同检查：

```bash
FORGE_LEROBOT_RUNTIME_ROOT=/absolute/path/to/forge_lerobot_runtime \
  dist/lerobot_policy/bin/check-policy

FORGE_LEROBOT_RUNTIME_ROOT=/absolute/path/to/forge_lerobot_runtime \
TORCH_HOME=/absolute/path/to/torchvision_resnet18 \
  dist/lerobot_policy/bin/lerobot --version
```

这里的 Dora CLI 由独立的 `dora_runtime` Resource 提供，不属于 Policy 或 LeRobot Runtime。发布或组合 Policy Resource 前，应使用 Runtime 自带的兼容性清单检查独立 `uv.lock` 的核心版本：

```bash
uv run python scripts/check_runtime_compatibility.py \
  /absolute/path/to/forge_lerobot_runtime/runtime-manifest.json
```

构建布局、显式 Runtime 绑定和 fake-runtime smoke 可通过以下命令验证：

```bash
bash scripts/test_thin_policy.sh
```
