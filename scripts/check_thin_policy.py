from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

EXPECTED_VERSION = "1.0.2"
FORBIDDEN = ("torch", "lerobot", "numpy", "cv2", "forge")


def _is_forbidden_entry(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    if normalized.startswith("lerobot_inference"):
        return False
    return normalized.startswith(("torch", "forge", "numpy", "cv2", "opencv", "lerobot"))


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _module_origin(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin in (None, "built-in", "frozen"):
        return None
    return _resolved(spec.origin)


def main() -> int:
    policy_root = _resolved(Path(__file__).parents[1])
    site_packages = policy_root / "site-packages"
    runtime_value = os.environ.get("FORGE_LEROBOT_RUNTIME_ROOT")
    if not runtime_value:
        print(
            "ERROR: 未设置 FORGE_LEROBOT_RUNTIME_ROOT；该路径必须由 Resource Resolver 显式提供。",
            file=sys.stderr,
        )
        return 1
    runtime_root = _resolved(runtime_value)

    errors: list[str] = []
    executable = _resolved(sys.executable)
    if not _is_relative_to(executable, runtime_root):
        errors.append(f"runtime Python 来源错误: {executable}（期望位于 {runtime_root}）")

    lerobot_inference = __import__("lerobot_inference")
    policy_file = getattr(lerobot_inference, "__file__", None)
    policy_version = getattr(lerobot_inference, "__version__", None)
    if policy_file is None:
        errors.append("policy 模块没有可验证的 __file__")
        policy_origin = site_packages / "<unknown>"
    else:
        policy_origin = _resolved(policy_file)
        if not _is_relative_to(policy_origin, site_packages):
            errors.append(f"policy 模块来源错误: {policy_origin}（期望位于 {site_packages}）")
    if policy_version != EXPECTED_VERSION:
        errors.append(f"policy 版本错误: {policy_version!r}（期望 {EXPECTED_VERSION!r}）")

    forbidden_entries = sorted(
        entry.name
        for entry in site_packages.iterdir()
        if _is_forbidden_entry(entry.name)
    )
    if forbidden_entries:
        errors.append("Policy site-packages 含禁止依赖: " + ", ".join(forbidden_entries))

    checked_runtime_modules: list[str] = []
    for name in FORBIDDEN:
        origin = _module_origin(name)
        if origin is None:
            continue
        if _is_relative_to(origin, site_packages):
            errors.append(f"禁止模块 {name} 来自 Policy: {origin}")
        elif not _is_relative_to(origin, runtime_root):
            errors.append(f"runtime 模块 {name} 来源错误: {origin}（期望位于 {runtime_root}）")
        else:
            checked_runtime_modules.append(f"{name}={origin}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: runtime={executable}")
    print(f"OK: policy={policy_origin} version={policy_version}")
    if checked_runtime_modules:
        print("OK: runtime modules: " + ", ".join(checked_runtime_modules))
    print("OK: Policy site-packages 未携带禁止依赖")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
