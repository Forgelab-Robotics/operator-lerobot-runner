# Inference configuration examples

每个已实现的策略都有独立配置，不需要在一个通用 YAML 中手动切换注释。

| 配置 | 策略 / backend | 用途 |
| --- | --- | --- |
| `act.yaml` | ACT sync / async chunked | 同步 `select_action()`（可选 temporal ensemble）或后台非阻塞 action chunk |
| `pi05.yaml` | PI0.5 sync | LeRobot 原生 `select_action()` chunk queue |
| `pi05_async_rtc.yaml` | PI0.5 async RTC | 后台 chunk 推理和非阻塞 Real-Time Chunking |
| `smoke.yaml` | ACT | 验证 ACT 转换产物的本地 `infer-once` 冒烟配置 |

示例：

```bash
uv run lerobot infer-once --config config/inference/act.yaml
uv run lerobot infer-once --config config/inference/pi05.yaml
uv run lerobot infer --config config/inference/pi05_async_rtc.yaml
```

使用前必须修改模型路径，并确保：

- `joints`、可选的 `state_joints` 与 checkpoint 的 state/action features 一致。
- `image_inputs` 的 key 可使用任意 Dora image topic/input ID，alias 用于匹配 checkpoint 的 `observation.images.<alias>`。ACT 和 PI0.5 都使用运行时与 checkpoint 相机的交集（至少一路匹配），未匹配的额外 topic 会被忽略；PI0.5 会将缺失视角补为 masked empty image。
- ACT 默认 `inference_mode: sync`；`async_chunked` 默认 `control_hz: 30`、`chunk_size_threshold: 0.5`，`actions_per_chunk` 省略时读取 checkpoint 的 `n_action_steps`。`temporal_ensemble_coeff` 仅用于 sync。
- PI0.5 的 `tokenizer_path` 指向完整的本地 tokenizer 目录。
- PI0.5 的 `instruction` 与训练数据中的 task 文本一致。
- YAML 中的相对路径相对于该 YAML 所在目录解析。

完整 Dora 和仿真 dataflow 见项目 `examples/` 目录。
