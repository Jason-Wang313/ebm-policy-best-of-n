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
$PYTHON experiments/run_controlled_energy_tail.py
$PYTHON experiments/run_learned_ibc_policy.py
$PYTHON experiments/run_learned_torch_ibc.py
$PYTHON experiments/run_multimodal_action_energy.py
$PYTHON experiments/run_action_optimization_budget.py
$PYTHON experiments/run_repair_and_gates.py
$PYTHON experiments/run_calibration_ablation.py
$PYTHON experiments/run_exact_law_validation.py
$PYTHON experiments/run_metaworld_benchmark.py
$PYTHON experiments/run_optional_benchmarks.py
$PYTHON scripts/run_figures.py
