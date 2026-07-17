# examples/

真机推理示例对齐 [`single_real_inference`](../single_real_inference/)（相机 / Piper / task_robot / 录制），策略节点为本仓库统一入口 `lerobot infer`（或打包二进制 `… infer --config …`）。

| 目录 | 说明 |
|------|------|
| [dora_convert](./dora_convert/) | policy_train → LeRobot 格式 |
| [dora_infer_act_7d](./dora_infer_act_7d/) | 单臂 7D 真机（对标 single_real_inference） |
| [dora_infer_act_14d](./dora_infer_act_14d/) | 双臂 14D 真机 |
| [dora_infer_act](./dora_infer_act/) | 索引 |

```text
joints 个数 == 模型 state/action 维数 == 机器人自由度
```

`dataflow.yaml` 中 forge 节点默认指向同级 `../../../forge_runtime/bin/...`（源码 `main.py` 已注释）；策略节点用本仓库 `../../bin/lerobot_infer/lerobot_infer`（需先 `bash scripts/build.sh`）。
