# Inference configuration examples

每个已实现的策略都有独立配置，不需要在一个通用 YAML 中手动切换注释。

| 配置 | 策略 / backend | 用途 |
| --- | --- | --- |
| `act.yaml` | ACT | 标准 action chunk，可选 temporal ensemble |
| `pi05.yaml` | PI0.5 sync | LeRobot 原生 `select_action()` chunk queue |
| `pi05_async_rtc.yaml` | PI0.5 async RTC | 后台 chunk 推理和非阻塞 Real-Time Chunking |
| `smoke.yaml` | ACT | 验证 ACT 转换产物的本地 `infer-once` 冒烟配置 |
| `vla_jepa.yaml` | VLA-JEPA | 指令条件 VLA 推理；官方 LIBERO 双相机 checkpoint |
| `../examples/libero_eval/policy_vla_jepa.yaml` | VLA-JEPA + LIBERO | 真实 LIBERO 闭环评测 |

示例：

```bash
uv run lerobot infer-once --config config/inference/act.yaml
uv run lerobot infer-once --config config/inference/pi05.yaml
uv run lerobot infer --config config/inference/pi05_async_rtc.yaml
uv run lerobot infer-once --config config/inference/vla_jepa.yaml
```

使用前必须修改模型路径，并确保：

- `joints`、可选的 `state_joints` 与 checkpoint 的 state/action features 一致。
- `image_inputs` 的 alias 与 checkpoint 的 `observation.images.<alias>` 一致。
- PI0.5 的 `tokenizer_path` 指向完整的本地 tokenizer 目录。
- PI0.5 的 `instruction` 与训练数据中的 task 文本一致。
- YAML 中的相对路径相对于该 YAML 所在目录解析；LIBERO 示例默认使用仓库同级的 `../LIBERO`。

完整 Dora 和仿真 dataflow 见项目 `examples/` 目录。
