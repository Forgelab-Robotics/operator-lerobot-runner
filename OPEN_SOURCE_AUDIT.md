# Open-source release audit

Audit date: 2026-09-03

## Decision

The audited release is suitable for publication with its existing Git history,
branches, and tags. Project-owned code, configuration, static resources, and
documentation use Apache-2.0. Runtime and build dependencies resolve entirely
from public indexes and retain the licenses declared by their distributions,
as documented in `THIRD_PARTY_NOTICES.md`.

## Completed checks

- `uv sync --frozen --extra dev` resolves all dependencies from public
  indexes; no path, Git, or private-index sources remain.
- Forge runtime dependencies resolve from PyPI: `forge-common==1.0.1`,
  `forge-policy==1.0.1`, `forge-tool[dora]==1.0.0`, `forge-msgs==1.2.0`.
  All four import cleanly in the locked environment.
- All 156 tests pass on Python 3.12 without physical hardware.
- `uv run ruff check .` is clean. `TRY004` is config-ignored (config
  validation deliberately raises `ValueError` for end users); deliberate
  blind-exception worker/stop-hook boundaries carry inline `noqa: BLE001`
  comments with justification.
- CLI verifies: `--help` succeeds and `--version` reports `lerobot 1.0.5`.
- `pip-audit` on the locked runtime export reports no actionable findings:
  4 findings are accepted as documented known limitations (below) and are
  ignored by ID in the CI dependency-audit job, so any new vulnerability
  still fails CI.
- `detect-secrets` reports zero findings in the publishable source tree.
- A `detect-secrets` scan across every historical Git blob (348 blobs,
  all objects in the repository) reports zero findings.
- Private repository URLs and machine-specific paths
  (`gitlab.ex-ai.cn`, `meta-emt`, `/home/`, `/Users/`) are absent from the
  current publishable tree.
- No file approaches GitHub's 100 MiB hard limit; the largest tracked file
  is `uv.lock` at 89,845 bytes (~88 KiB).
- GitHub Actions CI (test / dependency-audit / secrets), Dependabot, issue,
  pull-request, contribution, and security policy files are included.

## License findings

- Project-owned Python, configuration, static resources, tests, and
  documentation: Apache-2.0.
- No third-party runtime component is vendored; runtime dependencies retain
  the licenses declared by their distributions (see
  `THIRD_PARTY_NOTICES.md`).
- Torch (BSD-3-Clause) and TorchVision (BSD-3-Clause) are pinned to the
  stable CUDA 13 wheels (`torch 2.11.0+cu130`, `torchvision 0.26.0+cu130`)
  from the default PyPI index, aligned with the `lerobot_trainer` core
  versions documented in the README.
- Triton: MIT. LeRobot, Transformers, Diffusers, Safetensors,
  Hugging Face Hub, OpenCV Python wheels, and the Forge packages:
  Apache-2.0. NumPy: BSD-3-Clause. Einops and PyYAML: MIT. Pillow: HPND.
  Dora-rs: MIT.
- PyInstaller is a build-only dependency under GPL-2.0-or-later with its
  special exception; the PyInstaller-built binary distribution requires a
  separate artifact-level review.

## Known limitations

- `torch 2.11.0` is affected by PYSEC-2025-194 / CVE-2025-3000
  (`torch.jit.script` memory corruption). LeRobot 0.6.1 constrains
  `torch<2.12.0`, so an upgrade is not possible without breaking the
  dependency contract, and the torch pin is aligned with `lerobot_trainer`
  and validated on RTX 5060 hardware. The repository does not use
  `torch.jit.script` (compilation uses `torch.compile`), loads checkpoints
  with `weights_only=True`, and is intended to run only trusted model
  weights.
- `transformers 5.5.4` is affected by CVE-2026-9856 (path traversal in
  tokenizer/processor `save_pretrained()` via malicious `chat_template`
  keys from a Hub repository). LeRobot 0.6.1 constrains
  `transformers<5.6.0`. The repository loads pretrained tokenizers and
  processors read-only and does not save tokenizers from untrusted
  repositories.
- `setuptools 81.0.0` is affected by PYSEC-2026-3447 (macOS APFS/HFS+ file
  name normalization bypass when packing sdists). LeRobot 0.6.1 constrains
  `setuptools<82.0.0`. The project publishes no source distribution and
  releases only on Linux x86_64; the exposure is not applicable to this
  release.
- CI and this audit do not command physical hardware. Runtime deployment is
  only supported on trusted networks; Dora dataflows must not be exposed
  directly to the public internet (see `SECURITY.md`).
- PyInstaller binary releases bundle NVIDIA user-space CUDA libraries at
  build time and require a separate artifact-level license and security
  review (see `THIRD_PARTY_NOTICES.md`).
- This audit is an engineering review, not legal advice, penetration
  testing, or safety certification.

## Publication model

The audited current tree is published together with the repository's
existing commit graph, branches, and tags. Historical commits are retained
for traceability and may contain obsolete internal repository locations;
they must not be treated as current installation instructions.
