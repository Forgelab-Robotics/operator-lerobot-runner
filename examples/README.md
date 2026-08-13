# examples/

真机推理示例对齐 [`single_real_inference`](../single_real_inference/)（相机 / Piper / task_robot / 录制），策略节点为本仓库统一入口 `lerobot infer`（或打包二进制 `… infer --config …`）。

| 目录 | 说明 |
|------|------|
| [dora_convert](./dora_convert/) | policy_train → LeRobot 格式 |
| [dora_infer_act_7d](./dora_infer_act_7d/) | 单臂 7D 真机（对标 single_real_inference） |
| [dora_infer_act_14d](./dora_infer_act_14d/) | 双臂 14D 真机 |
| [dora_sim_infer_act_7d_bin](./dora_sim_infer_act_7d_bin/) | 单臂 Piper MuJoCo + ACT 7D，全部节点使用本仓库 `bin/` |
| [dora_sim_infer_pi05_7d_bin](./dora_sim_infer_pi05_7d_bin/) | 单臂 Piper MuJoCo + PI0.5 7D，全部节点使用本仓库 `bin/` |
| [dora_sim_infer_fastwam_libero](./dora_sim_infer_fastwam_libero/) | FastWAM LIBERO 8D state/7D action、双图像 Forge/Dora 协议仿真 |
| [dora_sim_infer_fastwam_robotwin](./dora_sim_infer_fastwam_robotwin/) | FastWAM RoboTwin 14D、组合图像 Forge/Dora 协议仿真 |
| [dora_infer_act](./dora_infer_act/) | 索引 |

```text
默认配置：joints 对齐模型 action；state_joints 未单独配置时沿用 joints
```

`dataflow.yaml` 中 forge 节点默认指向同级 `../../../forge_runtime/bin/...`（源码 `main.py` 已注释）；策略节点用本仓库 `../../bin/lerobot_infer/lerobot_infer`（需先 `bash scripts/build.sh`）。

FastWAM 示例例外：它们使用本仓库 `.venv/bin/lerobot` 和共享的轻量 Forge 协议
模拟节点，验证消息路由、Policy 生命周期和 action 闭环，不作为 LIBERO/RoboTwin
物理任务或 Benchmark。运行时资源限制由硬件运行环境统一管理；完成手动生命周期
和 action 闭环验证后，最后才运行 `dataflow_auto.yaml`。
