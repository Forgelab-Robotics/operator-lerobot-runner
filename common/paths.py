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
    override = os.environ.get("LEROOT_LAUNCH_CWD")
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            return path.resolve()

    cwd = Path.cwd().resolve()
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass or cwd != Path(meipass).resolve():
        # Dora changes the subprocess cwd but may leave the inherited PWD value
        # untouched, so the actual process cwd is authoritative.
        return cwd

    preserved_pwd = os.environ.get("PWD")
    if preserved_pwd:
        path = Path(preserved_pwd).expanduser()
        if path.is_dir():
            return path.resolve()
    return cwd


def resolve_user_path(path: str | Path) -> Path:
    """Resolve a path passed by the user.

    - Absolute paths: resolved as-is.
    - Relative paths: against :func:`launch_cwd`, never against ``_internal`` / ``_MEIPASS``.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (launch_cwd() / p).resolve()
