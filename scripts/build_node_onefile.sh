#!/usr/bin/env bash
# Build the policy runner as a single-file (onefile) node payload.
#
# This is an optional packaging variant next to the default onedir build
# (scripts/build.sh); inference/session code and model assets are unchanged.
#
# Usage:
#   bash scripts/setup.sh
#   bash scripts/build_node_onefile.sh
#
# Optional:
#   PYTHON=/path/to/python bash scripts/build_node_onefile.sh
#   EXTRA_PYINSTALLER_ARGS='--log-level DEBUG' bash scripts/build_node_onefile.sh
#
# 产物：
#   dist/onefile/lerobot_infer
#
# PAOS executable_tar_gz 契约要求归档根目录仅包含一个名为 entrypoint 的普通文件，
# 因此打包命令为（归档内只有一行 lerobot_infer）：
#   tar -czf lerobot_infer-<version>-linux-x86_64.tar.gz -C dist/onefile lerobot_infer
#
# 本变体使用独立的 dist/build 子目录，不影响 scripts/build.sh 的 onedir 产物。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      sed -n '2,24p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: ${arg}" >&2
      exit 1
      ;;
  esac
done

# 构建解释器：默认使用仓库内 .venv，可用 PYTHON 显式覆盖；不回退到系统 Python。
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: 未找到构建解释器 ${PYTHON}" >&2
  echo "       请先执行: bash scripts/setup.sh" >&2
  echo "       或显式指定: PYTHON=/path/to/python bash scripts/build_node_onefile.sh" >&2
  exit 1
fi

SPEC="${SCRIPT_DIR}/lerobot_infer_node.spec"
ENTRY="${SCRIPT_DIR}/_pyi_entry_infer.py"
RTHOOK_COMPILE="${SCRIPT_DIR}/pyi_rth_torch_compile.py"
RTHOOK_NVIDIA="${SCRIPT_DIR}/pyi_rth_nvidia_preload.py"
DIST_DIR="${ROOT}/dist/onefile"
DIST_BIN="${DIST_DIR}/lerobot_infer"
WORK="${ROOT}/build/pyinstaller/lerobot_infer_node"

for need in "${SPEC}" "${ENTRY}" "${RTHOOK_COMPILE}" "${RTHOOK_NVIDIA}"; do
  if [[ ! -f "${need}" ]]; then
    echo "ERROR: missing build input: ${need}" >&2
    exit 1
  fi
done

if ! "${PYTHON}" -c 'import PyInstaller' >/dev/null 2>&1; then
  echo "ERROR: PyInstaller 未安装在 ${PYTHON}" >&2
  echo "       请执行: uv sync --extra build" >&2
  exit 1
fi

if ! "${PYTHON}" - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in ("torch", "torchvision", "triton", "lerobot_inference")
    if importlib.util.find_spec(name) is None
]
if missing:
    sys.exit("missing packages: " + ", ".join(missing))
PY
then
  echo "ERROR: 构建环境缺少依赖，请先执行: bash scripts/setup.sh" >&2
  exit 1
fi

echo "==> project: ${ROOT}"
echo "==> python:  ${PYTHON}"
"${PYTHON}" -V

# 只清理本变体自己的产物，不触碰 scripts/build.sh 的 dist/lerobot_infer。
rm -rf "${DIST_DIR}" "${WORK}"
mkdir -p "${DIST_DIR}" "${WORK}"

PYINSTALLER_ARGS=(
  --noconfirm
  --clean
  --distpath "${DIST_DIR}"
  --workpath "${WORK}"
)
if [[ -n "${EXTRA_PYINSTALLER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  _extra_args=(${EXTRA_PYINSTALLER_ARGS})
  PYINSTALLER_ARGS+=("${_extra_args[@]}")
fi

echo "==> Building onefile node: lerobot_infer"
"${PYTHON}" -m PyInstaller "${PYINSTALLER_ARGS[@]}" "${SPEC}"

if [[ ! -f "${DIST_BIN}" || ! -x "${DIST_BIN}" ]]; then
  echo "ERROR: onefile executable was not produced: ${DIST_BIN}" >&2
  exit 1
fi

file "${DIST_BIN}"
"${DIST_BIN}" --help >/dev/null

echo
echo "Done."
echo "  onefile: ${DIST_BIN}"
echo
echo "打包为 PAOS executable_tar_gz 归档（归档根目录仅含 lerobot_infer）："
echo "  tar -czf lerobot_infer-<version>-linux-x86_64.tar.gz -C ${DIST_DIR} lerobot_infer"
