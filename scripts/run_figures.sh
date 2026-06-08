#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
if [ -x /mnt/c/Users/wangz/AppData/Local/Programs/Python/Python310/python.exe ]; then
  PYTHON=/mnt/c/Users/wangz/AppData/Local/Programs/Python/Python310/python.exe
elif [ -x /c/Users/wangz/AppData/Local/Programs/Python/Python310/python.exe ]; then
  PYTHON=/c/Users/wangz/AppData/Local/Programs/Python/Python310/python.exe
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON="/c/Windows/py.exe -3"
fi
$PYTHON scripts/run_figures.py
