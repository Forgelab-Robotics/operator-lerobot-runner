# dora_convert

用 [dora-rs](https://dora-rs.ai/) 一次性跑完 **policy_train → LeRobot** 转换。

`dataflow.yaml` 只起一个节点，参数在同目录 `convert.yaml`。

## 前置

```bash
cd /path/to/lerobot_inference
uv sync
```

编辑 `convert.yaml`：

- `src_dir` → `test_convert/ACT`
- `dst_dir` → `test_convert/out_put/converted`（**out_put 下指定子目录**，产物直接写在该目录）

无原始 `task.json` 时用同目录 `minimal_task.json`：**三路相机** `left/right/top`，ACT 超参对齐 `pick_and_place`。

## 用 Dora 启动

```bash
cd lerobot_inference/examples/dora_convert
dora run dataflow.yaml
```

节点走本项目的 `.venv/bin/lerobot`（首参 `convert`；与 `uv run lerobot convert` 同一环境）。

## 等价 CLI（不经 Dora）

```bash
cd lerobot_inference
uv run lerobot convert --config examples/dora_convert/convert.yaml
```

`convert.yaml` 里相对路径相对**该 yaml 所在目录**解析。

```text
src_dir → ../../test_convert/ACT
dst_dir → ../../test_convert/out_put/converted
```

干跑：

```bash
uv run lerobot convert --config examples/dora_convert/convert.yaml --dry-run
```

同一二进制：

```bash
bin/lerobot_infer/lerobot_infer convert --config examples/dora_convert/convert.yaml
```

## 输出

`dst_dir` 即推理可用的预训练目录（无 `checkpoints/` 嵌套）：

```text
test_convert/out_put/converted/
  model.safetensors
  config.json
  policy_preprocessor.json
  policy_postprocessor.json
  conversion_meta.json
```

转换完成后，把 `policy.pretrained_path` 指到该目录，再用 [dora_infer_act_14d](../dora_infer_act_14d/) 或 [dora_infer_act_7d](../dora_infer_act_7d/) 推理。
