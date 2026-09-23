#!/usr/bin/env bash
# Keep the project virtual environment visible to Python when Dora starts a node.
#
# Dora 0.4.1 canonicalizes a node's executable path.  If the dataflow points
# directly at `.venv/bin/python` (a symlink), Python is therefore started using
# the base/Conda interpreter's real path and no longer discovers `.venv`'s
# pyvenv.cfg.  Starting it from this non-symlink wrapper preserves the virtual
# environment path passed to Python.

set -euo pipefail

support_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${support_dir}/../.." && pwd -P)"
project_python="${repo_root}/.venv/bin/python"

if [[ ! -x "${project_python}" ]]; then
    echo "错误：未找到项目 Python：${project_python}；请先在仓库根目录运行 uv sync --extra dev" >&2
    exit 127
fi

exec "${project_python}" "$@"
