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
$PYTHON experiments/run_controlled_energy_tail.py --smoke
$PYTHON experiments/run_learned_ibc_policy.py --smoke
$PYTHON experiments/run_learned_torch_ibc.py --smoke
$PYTHON experiments/run_exact_law_validation.py --smoke
$PYTHON experiments/run_repair_and_gates.py --smoke
$PYTHON experiments/run_calibration_ablation.py --smoke
$PYTHON experiments/run_action_optimization_budget.py --smoke
$PYTHON experiments/run_multimodal_action_energy.py --smoke
$PYTHON experiments/run_metaworld_benchmark.py --smoke
$PYTHON experiments/run_optional_benchmarks.py --smoke
$PYTHON scripts/run_figures.py
