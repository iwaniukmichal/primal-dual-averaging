from __future__ import annotations

import sys
from pathlib import Path


COMPARISON_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LASSO_CODE_DIR = PROJECT_ROOT / "lasso_logistic_sparsity" / "code"

sys.path.insert(0, str(LASSO_CODE_DIR))

import plot as lasso_plot


SYNTHETIC_DATASETS = [
    "synthetic_logistic_small_5d.csv",
    "synthetic_logistic_sparse_20d.csv",
    "synthetic_logistic_noisy_20d.csv",
    "synthetic_logistic_imbalanced_10d.csv",
    "synthetic_logistic_big_sparse_50d.csv",
    "synthetic_logistic_big_noisy_80d.csv",
]


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--results-dir",
            str(COMPARISON_DIR / "results"),
            "--out-dir",
            str(COMPARISON_DIR / "plots"),
        ]
    )
    for dataset in SYNTHETIC_DATASETS:
        sys.argv.extend(["--dataset", dataset])
    lasso_plot.main()
