# PyInstaller runtime hook: preload nvidia .so for frozen binaries (cu12 wheels).
from __future__ import annotations

import ctypes
import glob
import os
import sys


def _preload_nvidia_libs() -> None:
    if not getattr(sys, "frozen", False):
        return
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return
    for lib_dir in glob.glob(os.path.join(base, "nvidia", "*", "lib")):
        if not os.path.isdir(lib_dir):
            continue
        for so in sorted(os.listdir(lib_dir)):
            if ".so" in so:
                try:
                    ctypes.CDLL(os.path.join(lib_dir, so))
                except OSError:
                    pass
    for npp in glob.glob(os.path.join(base, "**", "libnppicc.so.12"), recursive=True):
        existing = os.environ.get("LD_PRELOAD", "")
        os.environ["LD_PRELOAD"] = f"{npp}:{existing}" if existing else npp
        break


_preload_nvidia_libs()
