from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ebm_best_of_n.plotting import regenerate_all_figures


if __name__ == "__main__":
    regenerate_all_figures("results")
