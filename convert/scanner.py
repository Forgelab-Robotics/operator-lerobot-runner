"""Discover policy_train checkpoint files for conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_STANDARD_PATTERN = re.compile(r"^policy_epoch_(\d+)_standard\.safetensors$")
_FLASH_EPOCH_PATTERN = re.compile(r"^policy_epoch_(\d+)\.safetensors$")


@dataclass(frozen=True)
class ConvertJob:
    """一次转换任务。

    输出永远写到 ``dst_root / output_rel``：
    - 能解析轮次：``{prefix}/epoch_XXXXXX/``（即使只有一个也不用 single）
    - 无法解析轮次的单文件：``{prefix}/single/``
    """

    epoch: int | None
    checkpoint_path: Path
    output_rel: str


def resolve_output_prefix(src_dir: Path, output_prefix: str | None = None) -> str:
    """子目录前缀；默认用 ``src_dir`` 目录名。"""
    if output_prefix is not None and str(output_prefix).strip():
        return str(output_prefix).strip().strip("/\\")
    return Path(src_dir).expanduser().resolve().name


def _single_output_rel(prefix: str) -> str:
    return f"{prefix}/single"


def _epoch_output_rel(prefix: str, epoch: int) -> str:
    return f"{prefix}/epoch_{epoch:06d}"


def _parse_epoch_from_name(name: str) -> int | None:
    """从文件名解析轮次；认 standard / flash 命名。"""
    match = _STANDARD_PATTERN.match(name)
    if match:
        return int(match.group(1))
    match = _FLASH_EPOCH_PATTERN.match(name)
    if match:
        return int(match.group(1))
    return None


def _job_for_checkpoint(prefix: str, path: Path) -> ConvertJob:
    epoch = _parse_epoch_from_name(path.name)
    if epoch is not None:
        return ConvertJob(
            epoch=epoch,
            checkpoint_path=path,
            output_rel=_epoch_output_rel(prefix, epoch),
        )
    return ConvertJob(
        epoch=None,
        checkpoint_path=path,
        output_rel=_single_output_rel(prefix),
    )


def list_standard_checkpoints(src_dir: Path) -> list[tuple[int, Path]]:
    src_dir = src_dir.expanduser().resolve()
    found: list[tuple[int, Path]] = []
    for path in sorted(src_dir.glob("policy_epoch_*_standard.safetensors")):
        match = _STANDARD_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found, key=lambda item: item[0])


def list_safetensors_files(src_dir: Path) -> list[Path]:
    src_dir = src_dir.expanduser().resolve()
    return sorted(path for path in src_dir.glob("*.safetensors") if path.is_file())


def _resolve_explicit_checkpoint(src_dir: Path, checkpoint: str) -> Path:
    candidate = src_dir / checkpoint
    if not candidate.is_file():
        candidate = Path(checkpoint).expanduser()
        if not candidate.is_file():
            candidate = src_dir / Path(checkpoint).name
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if candidate.suffix != ".safetensors":
        raise ValueError(f"Checkpoint must be a .safetensors file, got {candidate.name}")
    return candidate.resolve()


def discover_convert_jobs(
    src_dir: Path,
    *,
    checkpoint: str | None = None,
    output_prefix: str | None = None,
) -> list[ConvertJob]:
    """发现待转换检查点。

    规则：
    1. 指定 ``checkpoint`` → 只用该文件；能解析轮次则 ``{prefix}/epoch_XXXXXX/``，
       否则 ``{prefix}/single/``。
    2. 任意个 ``policy_epoch_*_standard.safetensors``（含仅 1 个）→ 全部转换，
       各写到 ``{prefix}/epoch_XXXXXX/``。
    3. 目录里只有一个无法解析轮次的 ``*.safetensors`` → ``{prefix}/single/``。
    4. 否则报错。

    ``prefix`` 默认取 ``src_dir`` 目录名，可用 ``output_prefix`` 覆盖。
    """
    src_dir = src_dir.expanduser().resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    prefix = resolve_output_prefix(src_dir, output_prefix)

    if checkpoint:
        path = _resolve_explicit_checkpoint(src_dir, checkpoint)
        return [_job_for_checkpoint(prefix, path)]

    standards = list_standard_checkpoints(src_dir)
    if standards:
        return [
            ConvertJob(
                epoch=epoch,
                checkpoint_path=path,
                output_rel=_epoch_output_rel(prefix, epoch),
            )
            for epoch, path in standards
        ]

    safetensors = list_safetensors_files(src_dir)
    if len(safetensors) == 1:
        return [_job_for_checkpoint(prefix, safetensors[0])]

    if not safetensors:
        raise FileNotFoundError(
            f"No .safetensors under {src_dir}. "
            "Expect policy_epoch_*_standard.safetensors, or a single .safetensors file."
        )

    flash_only = [
        path.name
        for path in safetensors
        if _FLASH_EPOCH_PATTERN.match(path.name) and not _STANDARD_PATTERN.match(path.name)
    ]
    hint = ""
    if flash_only and not standards:
        hint = (
            f" Found Flash training weights only ({', '.join(flash_only[:3])}"
            f"{'...' if len(flash_only) > 3 else ''}); use *_standard.safetensors for convert."
        )
    raise FileNotFoundError(
        f"Ambiguous checkpoints under {src_dir}: {[p.name for p in safetensors]}. "
        "Provide --checkpoint FILENAME, or keep a single .safetensors, "
        "or multiple policy_epoch_*_standard.safetensors."
        + hint
    )


def resolve_checkpoint_path(src_dir: Path, checkpoint: str | None) -> Path:
    """兼容旧接口：返回单个检查点路径（多轮时取最新 standard）。"""
    jobs = discover_convert_jobs(src_dir, checkpoint=checkpoint)
    if len(jobs) == 1:
        return jobs[0].checkpoint_path
    standards = list_standard_checkpoints(src_dir)
    if standards:
        return standards[-1][1]
    return jobs[-1].checkpoint_path
