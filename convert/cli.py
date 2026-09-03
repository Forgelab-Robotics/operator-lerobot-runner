"""CLI: convert policy_train checkpoints to lerobot_trainer format."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from lerobot_inference.common.paths import resolve_user_path
from lerobot_inference.convert.meta import load_policy_train_meta
from lerobot_inference.convert.scanner import discover_convert_jobs
from lerobot_inference.convert.writer import convert_policy_train_to_lerobot

_PATH_KEYS = ("src_dir", "dst_dir", "task_json", "checkpoint")


def _load_yaml_config(path: str | Path) -> tuple[dict, Path]:
    config_path = resolve_user_path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {path}")
    return data, config_path.parent


def _resolve_path_value(value: str | Path | None, *, base_dir: Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if base_dir is not None:
        return str((base_dir / path).resolve())
    return str(resolve_user_path(path))


def _resolve_settings(args) -> dict:
    settings: dict = {}
    config_base: Path | None = None
    if getattr(args, "config", None):
        settings, config_base = _load_yaml_config(args.config)
        for key in _PATH_KEYS:
            if key in settings and settings[key] not in (None, ""):
                # checkpoint may be a bare filename (not a path)
                if key == "checkpoint" and "/" not in str(settings[key]) and "\\" not in str(settings[key]):
                    continue
                settings[key] = _resolve_path_value(settings[key], base_dir=config_base)
        # output_prefix 是名字，不是路径
        if "output_prefix" in settings and settings["output_prefix"] in ("", None):
            settings.pop("output_prefix", None)

    for key, arg_name in (
        ("src_dir", "src_dir"),
        ("dst_dir", "dst_dir"),
        ("checkpoint", "checkpoint"),
        ("task_json", "task_json"),
        ("dry_run", "dry_run"),
        ("output_prefix", "output_prefix"),
    ):
        value = getattr(args, arg_name, None)
        if value not in (None, False):
            if key in ("src_dir", "dst_dir", "task_json"):
                settings[key] = _resolve_path_value(value, base_dir=None)
            else:
                settings[key] = value
    if not settings.get("src_dir") or not settings.get("dst_dir"):
        raise SystemExit("必须提供 --src-dir 与 --dst-dir（或通过 --config 指定）。")
    return settings


def run_convert(args) -> int:
    """执行转换；args 为 argparse Namespace（统一入口或本模块 main）。"""
    settings = _resolve_settings(args)
    src_dir = Path(settings["src_dir"])
    dst_root = Path(settings["dst_dir"])
    dry_run = bool(settings.get("dry_run", False))
    task_json = settings.get("task_json")
    output_prefix = settings.get("output_prefix")

    try:
        jobs = discover_convert_jobs(
            src_dir,
            checkpoint=settings.get("checkpoint"),
            output_prefix=output_prefix,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    reports = []
    for job in jobs:
        meta, stats = load_policy_train_meta(
            src_dir,
            checkpoint_path=job.checkpoint_path,
            task_json=task_json,
        )
        dst_dir = dst_root / job.output_rel
        report = convert_policy_train_to_lerobot(meta, stats, dst_dir, dry_run=dry_run)
        report["source_checkpoint_name"] = job.checkpoint_path.name
        report["output_rel"] = job.output_rel
        if job.epoch is not None:
            report["epoch"] = job.epoch
        reports.append(report)

    payload = reports[0] if len(reports) == 1 else reports
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    """兼容旧入口：转发到统一 CLI 的 convert 子命令。"""
    from lerobot_inference.cli import main_convert

    return main_convert()


if __name__ == "__main__":
    sys.exit(main())
