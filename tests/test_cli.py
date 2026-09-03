from __future__ import annotations

import sys

import pytest
from lerobot_inference import __version__
from lerobot_inference.cli import _normalize_argv, main


def test_top_level_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"lerobot {__version__}"


def test_onedir_alias_keeps_version_at_top_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lerobot_infer", "--version"])

    assert _normalize_argv(None) == ["--version"]
