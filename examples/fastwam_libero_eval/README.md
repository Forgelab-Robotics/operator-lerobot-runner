# FastWAM × LIBERO 单场景闭环评估

该示例使用当前仓库的 `FastWAMPolicyAdapter` 在真实 LIBERO MuJoCo 环境中运行
闭环评估。它不同于 `dora_sim_infer_fastwam_libero` 的协议模拟器，会执行真实场景、
检查任务成功条件并输出逐步动作和双相机视频。

## 环境

只使用当前仓库的 uv 环境。LIBERO 仿真依赖放在独立 extra 中，不安装 LIBERO
仓库自带的旧版 `requirements.txt`：

```bash
cd /path/to/lerobot_runner
uv sync --extra dev --extra libero-eval
```

评估器不会安装或修改 LIBERO，而是在启动时严格绑定 `--libero-root` 指定的源码、
BDDL、资产和初始状态。

## 运行默认场景

默认选择 `libero_spatial` 的 task 0：将盘子与 ramekin 之间的黑碗放到盘子上。

```bash
export LIBERO_ROOT=/path/to/LIBERO

MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --extra libero-eval lerobot eval-libero \
  --config examples/fastwam_libero_eval/policy_fastwam.yaml \
  --libero-root "${LIBERO_ROOT}" \
  --suite libero_spatial \
  --task-id 0 \
  --episodes 1
```

默认使用固定初始状态 0、环境 seed 0、10 个 settle steps、20 Hz 控制频率以及
最多 300 个动作步。任务失败但完整跑到 horizon 是有效评估，命令仍返回 0；只有
依赖、模型、仿真或写文件等技术错误返回非零。

每次运行在 `out/<timestamp>/` 下生成：

- `result.json`：任务、版本、成功状态、步数、规划次数和耗时汇总；
- `actions.jsonl`：每一步的 7D action、规划标记、动作范围和奖励；
- `rollout.mp4`：旋转到训练方向后的 agentview/wrist 并排视频；
- `runtime_config.yaml`：本次解析后的策略和仿真参数；
- `run.log`：完整运行日志。

可先检查命令和默认参数而不加载模型：

```bash
uv run lerobot eval-libero --help
```
