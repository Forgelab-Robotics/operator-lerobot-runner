from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_runtime_compatibility.py"
SPEC = importlib.util.spec_from_file_location("check_runtime_compatibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _lock() -> dict:
    return {
        "package": [
            {"name": name, "version": "1.2.3"}
            for name in MODULE.SHARED_PACKAGES
        ]
    }


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "id": "forge_lerobot_runtime",
        "version": "0.1.0",
        "packages": {name: "1.2.3" for name in MODULE.SHARED_PACKAGES},
    }


def test_compatible_shared_packages_pass():
    assert MODULE.compatibility_errors(_lock(), _manifest()) == []


def test_version_drift_is_reported():
    manifest = _manifest()
    manifest["packages"]["torch"] = "9.9.9"
    assert MODULE.compatibility_errors(_lock(), manifest) == [
        "shared package version mismatch: torch Policy=1.2.3 Runtime=9.9.9"
    ]


def test_missing_runtime_package_is_reported():
    manifest = _manifest()
    del manifest["packages"]["forge-policy"]
    assert MODULE.compatibility_errors(_lock(), manifest) == [
        "Runtime manifest missing shared package: forge-policy"
    ]
