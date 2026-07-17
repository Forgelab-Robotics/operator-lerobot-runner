#!/usr/bin/env bash
# Convert policy_train ACT artifacts to lerobot_trainer checkpoint layout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ $# -lt 2 && -z "${CONFIG:-}" ]]; then
  echo "Usage: $0 <policy_train_ckpt_dir> <lerobot_output_dir> [--checkpoint FILE] [--task-json PATH] [--dry-run]"
  echo "   or: CONFIG=config/convert/example.yaml $0"
  exit 1
fi

if [[ -n "${CONFIG:-}" ]]; then
  exec uv run lerobot convert --config "${CONFIG}" "$@"
fi

SRC_DIR="$1"
DST_DIR="$2"
shift 2

exec uv run lerobot convert \
  --src-dir "${SRC_DIR}" \
  --dst-dir "${DST_DIR}" \
  "$@"
