#!/usr/bin/env bash
# Shell smoke for build layout, launcher delegation, runtime override, and check-policy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
"${SCRIPT_DIR}/build_thin_policy.sh"

POLICY_ROOT="${ROOT}/dist/lerobot_policy"
FAKE_ROOT="$(mktemp -d)"
trap 'rm -rf "${FAKE_ROOT}"' EXIT
RUNTIME_ROOT="${FAKE_ROOT}/runtime/forge_lerobot"
mkdir -p "${RUNTIME_ROOT}/bin"
ln -s "${ROOT}/.venv/bin/python" "${RUNTIME_ROOT}/bin/python"

# Keep sys.executable below the fake runtime while emulating the real Runtime's
# controlled FORGE_POLICY_SITE_PACKAGES -> PYTHONPATH bridge.
rm "${RUNTIME_ROOT}/bin/python"
cp "${ROOT}/.venv/bin/python" "${RUNTIME_ROOT}/bin/python-real"
cat > "${RUNTIME_ROOT}/bin/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${FORGE_POLICY_SITE_PACKAGES:?missing FORGE_POLICY_SITE_PACKAGES}"
exec "$(dirname "$0")/python-real" "$@"
EOF
chmod 0755 "${RUNTIME_ROOT}/bin/python" "${RUNTIME_ROOT}/bin/python-real"

if env -u FORGE_LEROBOT_RUNTIME_ROOT "${POLICY_ROOT}/bin/check-policy" >/dev/null 2>&1; then
  echo "ERROR: check-policy 未拒绝缺失的 Runtime Resource 路径" >&2
  exit 1
fi
if env -u FORGE_LEROBOT_RUNTIME_ROOT "${POLICY_ROOT}/bin/lerobot" --version >/dev/null 2>&1; then
  echo "ERROR: launcher 未拒绝缺失的 Runtime Resource 路径" >&2
  exit 1
fi

FORGE_LEROBOT_RUNTIME_ROOT="${RUNTIME_ROOT}" "${POLICY_ROOT}/bin/check-policy"
TORCH_ASSET_ROOT="${FAKE_ROOT}/resources/torchvision_resnet18"
mkdir -p "${TORCH_ASSET_ROOT}/hub/checkpoints"
VERSION_OUTPUT="$(FORGE_LEROBOT_RUNTIME_ROOT="${RUNTIME_ROOT}" FORGE_RUN_DIR="${FAKE_ROOT}/run" TORCH_HOME="${TORCH_ASSET_ROOT}" "${POLICY_ROOT}/bin/lerobot" --version)"
if [[ "${VERSION_OUTPUT}" != "lerobot 1.0.3" ]]; then
  echo "ERROR: launcher version smoke failed: ${VERSION_OUTPUT}" >&2
  exit 1
fi
if [[ ! -d "${FAKE_ROOT}/run/lerobot_policy/cache" ]]; then
  echo "ERROR: launcher did not create writable cache under FORGE_RUN_DIR" >&2
  exit 1
fi

TOP_LEVEL="$(find "${POLICY_ROOT}/site-packages" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)"
EXPECTED=$'lerobot_inference\nlerobot_inference-1.0.3.dist-info'
if [[ "${TOP_LEVEL}" != "${EXPECTED}" ]]; then
  echo "ERROR: unexpected Policy site-packages entries:" >&2
  printf '%s\n' "${TOP_LEVEL}" >&2
  exit 1
fi

echo "OK: thin Policy shell smoke passed"
