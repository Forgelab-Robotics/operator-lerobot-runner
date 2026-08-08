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
RUNTIME_ROOT="$(cd "${RUNTIME_ROOT}" 2>/dev/null && pwd -P)" || {
  echo "ERROR: Forge LeRobot runtime 根目录不存在: ${RUNTIME_ROOT}" >&2
  exit 1
}
PYTHON="${RUNTIME_ROOT}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Forge LeRobot runtime Python 不存在或不可执行: ${PYTHON}" >&2
  echo "请检查 Resource Resolver 提供的 FORGE_LEROBOT_RUNTIME_ROOT。" >&2
  exit 1
fi

RUN_ROOT="${FORGE_RUN_DIR:-${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/forge}"
CACHE_ROOT="${RUN_ROOT}/lerobot_policy/cache"
mkdir -p "${CACHE_ROOT}/pycache" "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/huggingface"
if [[ -n "${TORCH_HOME:-}" ]]; then
  if [[ ! -d "${TORCH_HOME}" ]]; then
    echo "ERROR: TORCH_HOME 不是已物化的目录: ${TORCH_HOME}" >&2
    exit 1
  fi
  TORCH_HOME="$(cd "${TORCH_HOME}" && pwd -P)"
else
  TORCH_HOME="${CACHE_ROOT}/torch"
  mkdir -p "${TORCH_HOME}"
fi

export PYTHONNOUSERSITE=1
export FORGE_POLICY_SITE_PACKAGES="${POLICY_ROOT}/site-packages"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/pycache"
export FORGE_LEROBOT_CACHE_DIR="${CACHE_ROOT}/xdg"
export TORCH_HOME
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_OFFLINE="${FORGE_HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${FORGE_TRANSFORMERS_OFFLINE:-1}"

exec "${PYTHON}" -m lerobot_inference "$@"
