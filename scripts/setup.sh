#!/usr/bin/env bash
# Bootstrap the standalone lerobot_inference uv environment.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "==> Syncing lerobot_inference uv environment in ${ROOT}"
uv sync "$@"

echo
echo "Done. Examples:"
echo "  uv run lerobot --help"
echo "  uv run lerobot convert --config examples/dora_convert/convert.yaml --dry-run"
echo "  uv run lerobot convert --config config/convert/example.yaml"
echo "  uv run lerobot infer-once --config config/inference/smoke.yaml"
echo "  bash scripts/run_once.sh --config config/inference/example.yaml"
echo "  bash scripts/convert_policy_train.sh <src> <dst> --task-json <task.json>"
