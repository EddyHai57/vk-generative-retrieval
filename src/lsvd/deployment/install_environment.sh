#!/bin/sh
set -eu

python_command=python3
environment_directory=.venv-lsvd-cu128
verify_only=0
allow_cpu=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --python) python_command=$2; shift 2 ;;
        --environment) environment_directory=$2; shift 2 ;;
        --verify-only) verify_only=1; shift ;;
        --allow-cpu) allow_cpu=1; shift ;;
        *) printf '%s\n' 'Unknown environment installer option' >&2; exit 2 ;;
    esac
done

script_directory=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
if [ "$verify_only" -eq 0 ]; then
    "$python_command" -c 'import platform; assert platform.python_version() == "3.12.14", "Python 3.12.14 required"'
    if [ -e "$environment_directory" ]; then
        printf '%s\n' 'Choose a new environment directory' >&2
        exit 2
    fi
    "$python_command" -m venv "$environment_directory"
    python_command=$environment_directory/bin/python
    "$python_command" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    "$python_command" -m pip install -r "$script_directory/requirements.txt"
    "$python_command" -m pip freeze > "$environment_directory/installed-packages.txt"
fi

if [ "$allow_cpu" -eq 1 ]; then
    "$python_command" -m lsvd.check_environment --allow-cpu
else
    "$python_command" -m lsvd.check_environment
fi
