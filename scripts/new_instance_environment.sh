#!/usr/bin/env bash
# Prepared locally; execute only on Eddy-approved NEW instance tomorrow.
set -euo pipefail
if [[ $# != 3 || "$1" != "--execute-on-new-instance" ]]; then
  echo 'usage: new_instance_environment.sh --execute-on-new-instance CODE_ROOT NEW_ENV_ROOT' >&2
  exit 2
fi
code_root="$2"
env_root="$3"
[[ -d "$code_root/configs/new-instance" && ! -e "$env_root" && ! -L "$env_root" ]]
mkdir "$env_root"
# New copies; original committed lock and pyproject are never overwritten.
cp "$code_root/configs/new-instance/anchor-pyproject.toml" "$env_root/pyproject.toml"
cp "$code_root/configs/new-instance/anchor-uv.lock" "$env_root/uv.lock"
cd "$env_root"
uv sync --frozen --extra training --python 3.11.16
cd "$code_root"
"$env_root/.venv/bin/python" -B -m analysis.new_instance_prepare env-check --lock "$env_root/uv.lock"
