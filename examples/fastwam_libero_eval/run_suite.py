#!/usr/bin/env python3
"""按 YAML 协议串行执行一个完整的 FastWAM × LIBERO task suite 子集。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _load_protocol(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"评测配置必须是 YAML mapping: {config_path}")

    required = {
        "policy_config",
        "libero_root",
        "output_dir",
        "suite",
        "task_ids",
        "episodes_per_task",
        "init_state_id",
        "seed",
        "episode_length",
        "num_steps_wait",
        "control_freq",
        "observation_size",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"评测配置缺少字段: {', '.join(missing)}")

    task_ids = payload["task_ids"]
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids 必须是非空列表")
    normalized_task_ids = [int(task_id) for task_id in task_ids]
    if len(set(normalized_task_ids)) != len(normalized_task_ids):
        raise ValueError("task_ids 不得重复")
    payload["task_ids"] = normalized_task_ids

    for key in (
        "episodes_per_task",
        "episode_length",
        "control_freq",
        "observation_size",
    ):
        payload[key] = int(payload[key])
        if payload[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    for key in ("init_state_id", "num_steps_wait"):
        payload[key] = int(payload[key])
        if payload[key] < 0:
            raise ValueError(f"{key} 必须为非负整数")
    payload["seed"] = int(payload["seed"])
    payload["continue_on_error"] = bool(payload.get("continue_on_error", False))
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _latest_result(task_root: Path) -> Path | None:
    candidates = sorted(task_root.glob("*/result.json"))
    return candidates[-1] if candidates else None


def _update_summary(
    *,
    run_root: Path,
    protocol: dict[str, Any],
    task_results: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    completed = [item for item in task_results if item["status"] == "completed"]
    completed_episodes = sum(int(item.get("total_episodes") or 0) for item in completed)
    successes = sum(int(item.get("successes") or 0) for item in completed)
    expected_episodes = len(protocol["task_ids"]) * protocol["episodes_per_task"]
    summary = {
        "schema_version": 1,
        "status": status,
        "suite": protocol["suite"],
        "expected_tasks": len(protocol["task_ids"]),
        "episodes_per_task": protocol["episodes_per_task"],
        "expected_total_episodes": expected_episodes,
        "completed_tasks": len(completed),
        "completed_episodes": completed_episodes,
        "successes": successes,
        "success_rate": successes / completed_episodes if completed_episodes else None,
        "task_results": task_results,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def run(config_path: Path) -> int:
    config_path = config_path.expanduser().resolve()
    protocol = _load_protocol(config_path)
    policy_config = _resolve_path(config_path, str(protocol["policy_config"]))
    libero_root = _resolve_path(config_path, str(protocol["libero_root"]))
    output_root = _resolve_path(config_path, str(protocol["output_dir"]))

    if not policy_config.is_file():
        raise FileNotFoundError(f"找不到策略配置: {policy_config}")
    if not (libero_root / "libero" / "libero").is_dir():
        raise FileNotFoundError(f"无效的 LIBERO 根目录: {libero_root}")

    run_id = f"{protocol['suite']}_{len(protocol['task_ids']) * protocol['episodes_per_task']}_"
    run_id += datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / run_id
    suffix = 1
    while run_root.exists():
        run_root = output_root / f"{run_id}_{suffix:02d}"
        suffix += 1
    run_root.mkdir(parents=True)

    manifest = {
        "protocol_config": str(config_path),
        "policy_config": str(policy_config),
        "libero_root": str(libero_root),
        "suite": protocol["suite"],
        "task_ids": protocol["task_ids"],
        "episodes_per_task": protocol["episodes_per_task"],
        "init_state_ids": list(
            range(
                protocol["init_state_id"],
                protocol["init_state_id"] + protocol["episodes_per_task"],
            )
        ),
        "episode_seeds": list(
            range(protocol["seed"], protocol["seed"] + protocol["episodes_per_task"])
        ),
        "episode_length": protocol["episode_length"],
        "num_steps_wait": protocol["num_steps_wait"],
        "control_freq": protocol["control_freq"],
        "observation_size": protocol["observation_size"],
        "control_mode": "relative",
        "hard_reset": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "start_time": datetime.now().astimezone().isoformat(),
    }
    (run_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    task_results: list[dict[str, Any]] = []
    _update_summary(
        run_root=run_root,
        protocol=protocol,
        task_results=task_results,
        status="running",
    )

    print(f"评测输出目录: {run_root}", flush=True)
    print(
        f"开始 {protocol['suite']}: {len(protocol['task_ids'])} tasks × "
        f"{protocol['episodes_per_task']} episodes = "
        f"{len(protocol['task_ids']) * protocol['episodes_per_task']} episodes",
        flush=True,
    )

    had_error = False
    for task_id in protocol["task_ids"]:
        task_root = run_root / f"task_{task_id:02d}"
        command = [
            sys.executable,
            "-m",
            "lerobot_inference",
            "eval-libero",
            "--config",
            str(policy_config),
            "--libero-root",
            str(libero_root),
            "--suite",
            str(protocol["suite"]),
            "--task-id",
            str(task_id),
            "--episodes",
            str(protocol["episodes_per_task"]),
            "--init-state-id",
            str(protocol["init_state_id"]),
            "--seed",
            str(protocol["seed"]),
            "--episode-length",
            str(protocol["episode_length"]),
            "--num-steps-wait",
            str(protocol["num_steps_wait"]),
            "--control-freq",
            str(protocol["control_freq"]),
            "--observation-size",
            str(protocol["observation_size"]),
            "--output-dir",
            str(task_root),
        ]
        print(f"\n[task {task_id}] {' '.join(command)}", flush=True)
        return_code = subprocess.run(command, check=False).returncode
        result_path = _latest_result(task_root)
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path is not None
            else {}
        )
        aggregate = result.get("aggregate", {})
        task_results.append(
            {
                "task_id": task_id,
                "status": result.get("status", "error"),
                "return_code": return_code,
                "result_path": str(result_path) if result_path else None,
                "successes": aggregate.get("successes"),
                "total_episodes": aggregate.get("total_episodes"),
                "success_rate": aggregate.get("success_rate"),
            }
        )
        had_error = had_error or return_code != 0 or result.get("status") != "completed"
        _update_summary(
            run_root=run_root,
            protocol=protocol,
            task_results=task_results,
            status="error" if had_error else "running",
        )
        if had_error and not protocol["continue_on_error"]:
            break

    completed_episodes = sum(int(item.get("total_episodes") or 0) for item in task_results)
    expected_episodes = len(protocol["task_ids"]) * protocol["episodes_per_task"]
    final_status = (
        "completed" if not had_error and completed_episodes == expected_episodes else "error"
    )
    summary = _update_summary(
        run_root=run_root,
        protocol=protocol,
        task_results=task_results,
        status=final_status,
    )
    print(
        f"评测结束: status={final_status}, successes={summary['successes']}/"
        f"{summary['completed_episodes']}, summary={run_root / 'summary.json'}",
        flush=True,
    )
    return 0 if final_status == "completed" else 1


def main() -> int:
    default_config = Path(__file__).with_name("libero_spatial_100.yaml")
    parser = argparse.ArgumentParser(description="运行 FastWAM × LIBERO suite 评测")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"评测协议 YAML（默认 {default_config}）",
    )
    args = parser.parse_args()
    try:
        return run(args.config)
    except KeyboardInterrupt:
        print("评测被用户中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"评测启动失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
