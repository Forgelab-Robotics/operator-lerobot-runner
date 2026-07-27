from __future__ import annotations

from pathlib import Path

from lerobot_inference.common.paths import launch_cwd, resolve_user_path


def test_launch_cwd_prefers_actual_process_directory(monkeypatch, tmp_path: Path) -> None:
    inherited_pwd = tmp_path / "parent"
    process_cwd = tmp_path / "dora-node"
    inherited_pwd.mkdir()
    process_cwd.mkdir()
    monkeypatch.chdir(process_cwd)
    monkeypatch.setenv("PWD", str(inherited_pwd))
    monkeypatch.delenv("LEROOT_LAUNCH_CWD", raising=False)

    assert launch_cwd() == process_cwd.resolve()
    assert resolve_user_path("policy.yaml") == (process_cwd / "policy.yaml").resolve()


def test_explicit_launch_cwd_override_is_honored(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "override"
    override.mkdir()
    monkeypatch.setenv("LEROOT_LAUNCH_CWD", str(override))

    assert launch_cwd() == override.resolve()
