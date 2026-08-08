#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_PACKAGES = (
    "torch",
    "torchvision",
    "lerobot",
    "transformers",
    "numpy",
    "opencv-python-headless",
    "dora-rs",
    "forge-common",
    "forge-msgs",
    "forge-policy",
)


def locked_versions(lock: dict[str, Any]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in lock.get("package", []):
        name = package.get("name")
        version = package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            versions[name] = version
    return versions


def compatibility_errors(
    lock: dict[str, Any], runtime_manifest: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if runtime_manifest.get("schema_version") != 1:
        errors.append("unsupported runtime manifest schema")
    if runtime_manifest.get("id") != "forge_lerobot_runtime":
        errors.append("runtime manifest id must be forge_lerobot_runtime")

    runtime_packages = runtime_manifest.get("packages")
    if not isinstance(runtime_packages, dict):
        return [*errors, "runtime manifest packages must be an object"]

    policy_packages = locked_versions(lock)
    for name in SHARED_PACKAGES:
        policy_version = policy_packages.get(name)
        runtime_version = runtime_packages.get(name)
        if policy_version is None:
            errors.append(f"Policy lock missing shared package: {name}")
        elif not isinstance(runtime_version, str):
            errors.append(f"Runtime manifest missing shared package: {name}")
        elif policy_version != runtime_version:
            errors.append(
                f"shared package version mismatch: {name} "
                f"Policy={policy_version} Runtime={runtime_version}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the standalone Policy lock with a Forge LeRobot Runtime"
    )
    parser.add_argument("runtime_manifest", type=Path)
    parser.add_argument("--lock", type=Path, default=PROJECT_ROOT / "uv.lock")
    args = parser.parse_args()

    lock = tomllib.loads(args.lock.read_text(encoding="utf-8"))
    runtime_manifest = json.loads(
        args.runtime_manifest.read_text(encoding="utf-8")
    )
    errors = compatibility_errors(lock, runtime_manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        "OK: Policy lock 与 "
        f"{runtime_manifest['id']}@{runtime_manifest['version']} 核心依赖一致"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
