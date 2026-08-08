#!/usr/bin/env bash
# Build the Studio thin Policy payload. Runtime dependencies stay in runtime/forge_lerobot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_ROOT="${ROOT}/dist/lerobot_policy"
SITE_PACKAGES="${DIST_ROOT}/site-packages"
BUILD_ROOT="${ROOT}/build/thin_policy"
WHEEL_DIR="${BUILD_ROOT}/wheel"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: 未找到 ${PYTHON}，请先执行 uv sync --extra dev" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 未找到 uv" >&2
  exit 1
fi

rm -rf "${DIST_ROOT}" "${BUILD_ROOT}"
mkdir -p "${SITE_PACKAGES}" "${DIST_ROOT}/bin" "${WHEEL_DIR}"

echo "==> Building lerobot-inference 1.0.2 wheel"
uv build --wheel --out-dir "${WHEEL_DIR}" "${ROOT}"
WHEELS=("${WHEEL_DIR}"/lerobot_inference-1.0.2-*.whl)
if [[ ! -f "${WHEELS[0]}" || "${#WHEELS[@]}" -ne 1 ]]; then
  echo "ERROR: 期望得到唯一的 lerobot-inference 1.0.2 wheel" >&2
  exit 1
fi

echo "==> Installing wheel without dependencies"
uv pip install --python "${PYTHON}" --no-deps --target "${SITE_PACKAGES}" "${WHEELS[0]}"
# uv places wheel console scripts below the target. The thin bundle has fixed launchers
# in Policy bin/, so this generated directory is redundant and violates the strict
# site-packages top-level contract.
rm -rf "${SITE_PACKAGES}/bin" "${SITE_PACKAGES}/.lock"

shopt -s nullglob dotglob
ENTRIES=("${SITE_PACKAGES}"/*)
for entry in "${ENTRIES[@]}"; do
  name="$(basename "${entry}")"
  case "${name}" in
    lerobot_inference|lerobot_inference-1.0.2.dist-info) ;;
    *)
      echo "ERROR: Policy site-packages 出现非 Runner 顶层内容: ${name}" >&2
      exit 1
      ;;
  esac
done
if [[ ! -d "${SITE_PACKAGES}/lerobot_inference" || ! -d "${SITE_PACKAGES}/lerobot_inference-1.0.2.dist-info" ]]; then
  echo "ERROR: wheel 安装内容不完整" >&2
  exit 1
fi

install -m 0755 "${SCRIPT_DIR}/thin_policy_launcher.sh" "${DIST_ROOT}/bin/lerobot"
install -m 0644 "${SCRIPT_DIR}/check_thin_policy.py" "${DIST_ROOT}/bin/check_thin_policy.py"
cat > "${DIST_ROOT}/bin/check-policy" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${FORGE_LEROBOT_RUNTIME_ROOT:-}"
if [[ -z "${RUNTIME_ROOT}" ]]; then
  echo "ERROR: 未设置 FORGE_LEROBOT_RUNTIME_ROOT；该路径必须由 Resource Resolver 显式提供。" >&2
  exit 1
fi
if [[ "${RUNTIME_ROOT}" != /* ]]; then
  echo "ERROR: FORGE_LEROBOT_RUNTIME_ROOT 必须是绝对路径: ${RUNTIME_ROOT}" >&2
  exit 1
fi
PYTHON="${RUNTIME_ROOT}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Forge LeRobot runtime Python 不存在或不可执行: ${PYTHON}" >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export FORGE_POLICY_SITE_PACKAGES="${POLICY_ROOT}/site-packages"
exec "${PYTHON}" "${POLICY_ROOT}/bin/check_thin_policy.py"
EOF
chmod 0755 "${DIST_ROOT}/bin/check-policy"

echo "==> Thin Policy ready: ${DIST_ROOT}"
echo "    Resource: lerobot_inference_policy@1.0.2"
echo "    Runtime 由 FORGE_LEROBOT_RUNTIME_ROOT 显式绑定"
