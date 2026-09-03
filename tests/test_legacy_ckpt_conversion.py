from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lerobot_inference.convert.scanner import discover_convert_jobs
from lerobot_inference.convert.writer import load_source_state_dict


def test_discovers_single_legacy_ckpt(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    torch.save({"weight": torch.ones(2)}, checkpoint)

    jobs = discover_convert_jobs(tmp_path)

    assert len(jobs) == 1
    assert jobs[0].checkpoint_path == checkpoint
    assert jobs[0].output_rel == f"{tmp_path.name}/single"


def test_explicit_legacy_ckpt_is_supported(tmp_path: Path) -> None:
    checkpoint = tmp_path / "selected.ckpt"
    torch.save({"weight": torch.ones(2)}, checkpoint)

    jobs = discover_convert_jobs(tmp_path, checkpoint=checkpoint.name, output_prefix="legacy")

    assert jobs[0].checkpoint_path == checkpoint
    assert jobs[0].output_rel == "legacy/single"


def test_legacy_ckpt_loads_tensor_only_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    torch.save({"state_dict": {"weight": torch.arange(3)}}, checkpoint)

    state_dict = load_source_state_dict(checkpoint)

    torch.testing.assert_close(state_dict["weight"], torch.arange(3))


def test_legacy_ckpt_rejects_non_tensor_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    torch.save({"metadata": "not a state dict"}, checkpoint)

    with pytest.raises(TypeError, match="tensor-only state_dict"):
        load_source_state_dict(checkpoint)
