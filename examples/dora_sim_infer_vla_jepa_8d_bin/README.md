# LIBERO + VLA-JEPA 8D（Dora 推理示例）

官方 `lerobot/VLA-JEPA-LIBERO` 策略的 Dora 推理示例（VLA-JEPA = Qwen3-VL
骨干 + Flow-matching DiT 动作头；V-JEPA2 世界模型仅训练使用，推理不执行）。

```text
libero_sim → task_robot → lerobot_infer(vla_jepa) → task_robot → libero_sim
                  └────────────── image_viewer ───────────────┘
```

## 1. 模型来源与许可证（完成标准记录）

| 组件 | 来源 | 仓库 | 许可证 |
|------|------|------|--------|
| Policy checkpoint | [lerobot/VLA-JEPA-LIBERO](https://huggingface.co/lerobot/VLA-JEPA-LIBERO) | official LeRobot 0.6 格式（config.json + model.safetensors + pre/postprocessor） | Apache-2.0 |
| Qwen3-VL 骨干 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 由 checkpoint `qwen_model_name` 引用，离线部署时本地化 | Apache-2.0 |
| V-JEPA2 编码器 | [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | 由 checkpoint `jepa_encoder_name` 引用；**推理不需要**，可 `enable_world_model: false` 跳过 | MIT |

**固定 revision 要求**：部署前记录各模型的实际 revision（`huggingface-cli` 或
Hub 页面），权重下载到本地目录，避免部署时依赖网络。本机已验证可用的
本地权重路径见下文。

## 2. 绑定模型（本地路径）

```bash
cd examples/dora_sim_infer_vla_jepa_8d_bin
ln -s /path/to/vla_jepa_libero_pretrained model
```

模型必须匹配：

- state：8D（7 关节角 + gripper 开合）；action：7D（7 关节）
- image features：`observation.images.image`、`observation.images.image2`
- LeRobot 0.6 processor schema（含归一化统计文件）

`model`、`logs/`、`out/` 均被 Git 忽略，不会提交机器相关路径。

## 3. 指令（instruction）

LIBERO 示例任务原句（不要改写为同义句，模型按训练分布生成）：

```yaml
instruction: "Place the red cube on top of the green cube"
```

运行时可通过 `adapter.instruction` 更新；更新会清空动作队列，旧指令下的
动作不会复用。

## 4. 冒烟测试（单次推理，可独立跑通）

```bash
# 源码环境（conda lerobot + PYTHONPATH 包布局）
uv run lerobot infer-once --config ./policy_vla_jepa.yaml

# 打包二进制（bash scripts/build.sh 后）
../../bin/lerobot_infer/lerobot_infer infer-once --config ./policy_vla_jepa.yaml
```

离线部署时取消 `policy_vla_jepa.yaml` 中 qwen_model_name / jepa_encoder_name
注释，指向本地模型目录（本机验证过的路径见文件内注释）。

## 5. adapter 可用性演示（无需仿真，可独立复现）

本节的命令**不依赖任何仿真/benchmark**：加载 VLA-JEPA adapter、用形状匹配的
假观测跑一步推理，输出动作 JSON。用于证明 adapter 可用（能加载、能推理、
输出形状正确）。

```bash
cd /mnt/SSD2_16TB/qinhan/lerobot_runner
export PYTHONPATH=/mnt/SSD2_16TB/qinhan/lr_pkg
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
python /mnt/SSD2_16TB/qinhan/lr_pkg/lerobot_inference/inference/run_once.py \
  --config examples/dora_sim_infer_vla_jepa_8d_bin/policy_vla_jepa.yaml \
  --height 224 --width 224
```

实测输出（2026-08-11，RTX 6000D）：模型加载约 16 s，之后单步推理出 7D 动作：

```text
(lerobot) qinhan@8xPro6000D-120:~/lerobot_runner$ python /mnt/SSD2_16TB/qinhan/lr_pkg/lerobot_inference/inference/run_once.py --config examples/dora_sim_infer_vla_jepa_8d_bin/policy_vla_jepa.yaml --height 224 --width 224
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
Loading weights: 100%|███████████████████████████████████████████████████████| 625/625 [00:00<00:00, 10041.22it/s]
Loading weights from local directory
WARNING:root:Unexpected key(s) when loading model: ['model.video_encoder.encoder.embeddings.patch_embeddings.proj.bias', ...]  # V-JEPA2 世界模型权重，推理用不到，见下文
{
  "policy_type": "vla_jepa",
  "pretrained_path": "/mnt/SSD2_16TB/qinhan/paos_deps/models/vla_jepa_libero",
  "action": [
    0.06480658054351807,
    0.013704061508178711,
    -0.02193915843963623,
    0.0018429458141326904,
    0.0032684504985809326,
    -0.001317441463470459,
    -1.0
  ],
  "action_dim": 7
}
```

> 注：`Unexpected key(s) ... model.video_encoder ...` 是加载器提示的
> **V-JEPA2 世界模型权重未被消费**（推理不需要，`enable_world_model: false`
> 跳过），不是错误，也不影响推理结果。

**为什么这能证明 adapter 可用**：

1. **官方入口**：`run_once.py` 内部走 `load_config` → `create_policy_adapter`
   （与 `lerobot infer` Dora 节点同一加载路径，非独立脚本）。
2. **真实 checkpoint**：`pretrained_path` 指向本地 `vla_jepa_libero`，625 个
   权重文件全部加载，无随机初始化回退。
3. **形状正确**：`action_dim: 7` 与 LIBERO 7 关节动作维度一致；第 7 维
   `-1.0` 为 gripper 关闭，符合训练动作分布。
4. **离线**：`HF_HUB_OFFLINE=1`，全程无网络访问（`Loading weights from local directory`）。

## 6. 资源占用（实测，RTX 6000D 85GB ×8）

`enable_world_model: false`（跳过 V-JEPA2，约省 2.6 GB 权重）实测：

| 指标 | 数值 |
|------|------|
| 模型加载耗时 | 16.0 s |
| 首次推理（生成 7 步 chunk 第 1 步） | 986 ms |
| GPU 显存峰值（allocated / reserved） | 6.18 / 6.23 GB |
| CPU 常驻内存峰值（VmHWM） | 8753 MB |

注：加载耗时与 CPU 内存包含 safetensors 权重映射与 Qwen tokenizer 初始化；
稳态推理时长为每次 `select_action`（约 chunk 中一步）的时间。8 GB 显存的
消费级 GPU 在模型初始化阶段可能 OOM，与 PI0.5 示例同结论，建议在
16 GB+ 显存环境运行。
