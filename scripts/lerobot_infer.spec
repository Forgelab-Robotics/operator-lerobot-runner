# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec：lerobot_infer onefile，启用 torch.compile（对齐 act-local-trainer）。"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

_spec_dir = Path(os.path.dirname(os.path.abspath(SPEC)))
_root_dir = _spec_dir.parent

datas: list = []
binaries: list = []
hiddenimports: list = []

# Preserve distribution metadata used by runtime dependency checks.
for dist in (
    "lerobot-inference", "lerobot", "diffusers", "transformers",
    "requests", "filelock", "numpy",
):
    datas += copy_metadata(dist, recursive=True)

for pkg in (
    "torch",
    "torchvision",
    "triton",
    "transformers",
    "diffusers",
    "lerobot",
    "safetensors",
    "einops",
    "cv2",
    "PIL",
    "yaml",
    "dora",
    "pyarrow",
):
    try:
        tmp = collect_all(pkg)
        datas += tmp[0]
        binaries += tmp[1]
        hiddenimports += tmp[2]
    except Exception as exc:
        raise RuntimeError(f"Failed to collect required package: {pkg}") from exc

hiddenimports += collect_submodules("lerobot_inference")
hiddenimports += collect_submodules("forge_msgs")
hiddenimports += collect_submodules("forge_common")
hiddenimports += collect_submodules("forge_policy")
hiddenimports += collect_submodules("forge_tool")
hiddenimports += [
    "lerobot_inference.cli",
    "lerobot_inference.convert.cli",
    "lerobot_inference.inference.main",
    "lerobot_inference.inference.run_once",
    "lerobot_inference.inference.compile_utils",
    "lerobot_inference.inference.policies.act",
    "lerobot_inference.inference.policies.pi05",
    "lerobot_inference.inference.policies.registry",
    "lerobot_inference.inference.policies.lingbot_va",
    "lerobot.policies.lingbot_va.modeling_lingbot_va",
    "torch._dynamo",
    "torch._dynamo.backends.inductor",
    "torch._functorch",
    "torch._higher_order_ops",
    "torch._inductor",
    "torch._inductor.compile_fx",
    "torch._inductor.codecache",
    "torch.fx",
    "triton",
]

# Triton JIT needs Python C headers under include/pythonX.Y
_py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
_include = sysconfig.get_path("include") or ""
_platinclude = sysconfig.get_path("platinclude") or ""
_include_dest = f"include/{_py_ver}"
if _include and os.path.isdir(_include):
    datas.append((_include, _include_dest))
if _platinclude and os.path.isdir(_platinclude) and _platinclude != _include:
    datas.append((_platinclude, _include_dest))

# Bundle NVIDIA wheel .so into root (skip libcuda — from host driver)
_site = Path(sysconfig.get_paths()["purelib"])
_nvidia = _site / "nvidia"
if _nvidia.is_dir():
    seen: set[str] = set()
    for so in _nvidia.glob("**/lib/*.so*"):
        if not so.is_file():
            continue
        name = so.name
        if name == "libcuda.so" or name.startswith("libcuda.so."):
            continue
        real = str(so.resolve())
        if real in seen:
            continue
        seen.add(real)
        binaries.append((str(so), "."))

runtime_hooks = [
    str(_spec_dir / "pyi_rth_nvidia_preload.py"),
    str(_spec_dir / "pyi_rth_torch_compile.py"),
]

a = Analysis(
    [str(_spec_dir / "_pyi_entry_infer.py")],
    pathex=[str(_root_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[
        "rerun_sdk",
        "IPython",
        "jupyter",
        "pytest",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

_EXE_ARGS = dict(
    name="lerobot_infer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# LEROBOT_PYI_MODE=onedir 时产出目录形态（对应 PAOS 的 directory_tar_gz）。
# onefile 的 CArchive 用 32 位偏移记录 TOC，内容超过 4 GiB 会直接构建失败；
# onedir 不生成 CArchive，没有这个上限。
if os.environ.get("LEROBOT_PYI_MODE") == "onedir":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **_EXE_ARGS)
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="lerobot_infer",
    )
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], **_EXE_ARGS)
