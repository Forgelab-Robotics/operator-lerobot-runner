"""PyInstaller runtime hook for torch.compile support.

Runs before the application imports torch. Mirrors act-local-trainer:
clear compile-disable flags and point Inductor/Triton caches to a writable dir.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_cache_root() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home)
    return Path.home() / ".cache"


# Do not inherit compile-disable flags from an older packaging environment.
os.environ.pop("TORCHDYNAMO_DISABLE", None)
os.environ.pop("TORCH_COMPILE_DISABLE", None)

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # Preserve the directory the user launched from; relative --config / yaml
    # paths must not resolve under _internal after we chdir for torch.compile.
    if "LEROOT_LAUNCH_CWD" not in os.environ:
        # Dora changes the child process cwd without necessarily updating the
        # inherited PWD environment variable.  getcwd() is the authoritative value.
        os.environ["LEROOT_LAUNCH_CWD"] = os.getcwd()
    os.chdir(sys._MEIPASS)

cache_root = _default_cache_root() / "lerobot_inference"
torchinductor_cache = cache_root / "torchinductor"
triton_cache = cache_root / "triton"
torchinductor_cache.mkdir(parents=True, exist_ok=True)
triton_cache.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torchinductor_cache))
os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
