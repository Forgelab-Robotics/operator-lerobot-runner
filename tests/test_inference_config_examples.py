from pathlib import Path

import pytest
from lerobot_inference.inference.config import PolicyNodeConfig
from lerobot_inference.inference.policies.registry import (
    normalize_policy_type,
    supported_policy_types,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "inference"


@pytest.mark.parametrize(
    ("filename", "expected_type", "expected_mode"),
    [
        ("act.yaml", "act", None),
        ("pi05.yaml", "pi05", "sync"),
        ("pi05_async_rtc.yaml", "pi05", "async_rtc"),
        ("fastwam_libero.yaml", "fastwam", None),
        ("fastwam_robotwin.yaml", "fastwam", None),
    ],
)
def test_inference_config_example_is_loadable(
    filename: str,
    expected_type: str,
    expected_mode: str | None,
) -> None:
    config = PolicyNodeConfig.from_yaml_path(_CONFIG_DIR / filename)

    assert normalize_policy_type(str(config.policy["type"])) == expected_type
    assert config.policy.get("inference_mode") == expected_mode
    assert config.joints
    assert config.image_input_id_to_alias


def test_every_implemented_policy_has_a_config_example(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DIFFUSION_POLICY_CHECKPOINT", str(tmp_path / "checkpoint"))

    example_types = {
        normalize_policy_type(
            str(PolicyNodeConfig.from_yaml_path(path).policy["type"])
        )
        for path in _CONFIG_DIR.glob("*.yaml")
    }

    assert set(supported_policy_types()) <= example_types


def test_diffusion_checkpoint_path_expands_environment_variable(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint"
    monkeypatch.setenv("TEST_DIFFUSION_CHECKPOINT", str(checkpoint))
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        """
joints: [joint1]
state_joints: [state1]
policy:
  type: diffusion
  pretrained_path: ${TEST_DIFFUSION_CHECKPOINT}
image_inputs:
  image/main: image
""".strip(),
        encoding="utf-8",
    )

    config = PolicyNodeConfig.from_yaml_path(config_path)

    assert config.policy["pretrained_path"] == str(checkpoint)


def test_diffusion_checkpoint_path_rejects_unset_environment_variable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MISSING_DIFFUSION_CHECKPOINT", raising=False)
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        """
joints: [joint1]
policy:
  type: diffusion
  pretrained_path: ${MISSING_DIFFUSION_CHECKPOINT}
image_inputs:
  image/main: image
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unset environment variables"):
        PolicyNodeConfig.from_yaml_path(config_path)


def test_diffusion_relative_checkpoint_path_remains_compatible(tmp_path: Path) -> None:
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        """
joints: [joint1]
policy:
  type: diffusion
  pretrained_path: checkpoints/model
image_inputs:
  image/main: image
""".strip(),
        encoding="utf-8",
    )

    config = PolicyNodeConfig.from_yaml_path(config_path)

    assert config.policy["pretrained_path"] == str(
        (tmp_path / "checkpoints/model").resolve()
    )


def test_other_policy_path_expansion_behavior_is_unchanged(monkeypatch, tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy-checkpoint"
    monkeypatch.setenv("LEGACY_LITERAL_PATH", str(legacy_path))
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(
        """
joints: [joint1]
policy:
  type: ACT
  pretrained_path: ${LEGACY_LITERAL_PATH}
image_inputs:
  image/main: image
""".strip(),
        encoding="utf-8",
    )

    config = PolicyNodeConfig.from_yaml_path(config_path)

    assert config.policy["pretrained_path"] == str(legacy_path)
