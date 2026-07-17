"""Resolve user-facing filesystem paths (CLI / YAML), especially under PyInstaller."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def launch_cwd() -> Path:
    """Working directory from which the user launched the process.

    PyInstaller runtime hooks may ``chdir`` into ``_MEIPASS``; prefer the preserved
    launch cwd / shell ``PWD`` so relative CLI paths still work.
    """
    for key in ("LEROOT_LAUNCH_CWD", "PWD"):
        value = os.environ.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.is_dir():
            return path.resolve()
    return Path.cwd().resolve()


def resolve_user_path(path: str | Path) -> Path:
    """Resolve a path passed by the user.

    - Absolute paths: resolved as-is.
    - Relative paths: against :func:`launch_cwd`, never against ``_internal`` / ``_MEIPASS``.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (launch_cwd() / p).resolve()
