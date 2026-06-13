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
run_step() {
  echo "==> $*"
  "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    if kill -0 "$pid" 2>/dev/null; then
      echo "... still running: $*"
    fi
  done
  wait "$pid"
  echo "<== $*"
}
run_step $PYTHON experiments/run_controlled_energy_tail.py --smoke
run_step $PYTHON experiments/run_learned_ibc_policy.py --smoke
run_step $PYTHON experiments/run_learned_torch_ibc.py --smoke
run_step $PYTHON experiments/run_exact_law_validation.py --smoke
run_step $PYTHON experiments/run_repair_and_gates.py --smoke
run_step $PYTHON experiments/run_calibration_ablation.py --smoke
run_step $PYTHON experiments/run_action_optimization_budget.py --smoke
run_step $PYTHON experiments/run_multimodal_action_energy.py --smoke
run_step $PYTHON experiments/run_metaworld_benchmark.py --smoke
run_step $PYTHON experiments/run_optional_benchmarks.py --smoke
run_step $PYTHON scripts/run_figures.py
