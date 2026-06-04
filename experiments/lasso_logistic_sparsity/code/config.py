from __future__ import annotations

from _shared.grids import parameter_grid


EXPERIMENT = "lasso_logistic_sparsity"
DESCRIPTION = "Compare lasso logistic sparsity and accuracy for custom solvers and sklearn L1 saga."


DATASETS = [
    "iris.csv",
    "synthetic_logistic_small_5d.csv",
    "synthetic_logistic_sparse_20d.csv",
    "synthetic_logistic_noisy_20d.csv",
    "synthetic_logistic_imbalanced_10d.csv",
    "breast_cancer_wdbc.csv",
    "banknote_authentication.csv",
    "spambase.csv",
]


def build_configs() -> list[dict[str, object]]:
    common = {
        "dataset": DATASETS,
        "lambda": [1.0],
        "restrict_to_fd": [False],
        "seed": [0],
        "test_size": [0.2],
        "D": [8.0, 32.0, 128.0],
        "max_iter": [1000],
        "eps": [1e-3],
        "lasso": [True],
    }
    configs: list[dict[str, object]] = []
    for dual_averaging in ["simple", "weighted"]:
        method = f"sda_{dual_averaging}_euclidean"
        for config in parameter_grid({**common, "gamma_mult": [0.25, 0.5, 1.0, 2.0]}):
            configs.append(
                {
                    **config,
                    "method": method,
                    "prox_fun": "euclidean",
                    "dual_averaging": dual_averaging,
                }
            )
    for config in parameter_grid({**common, "alpha": [0.1, 0.5, 1.0]}):
        configs.append({**config, "method": "projected_subgradient"})
    for config in parameter_grid(
        {
            "dataset": DATASETS,
            "lambda": [1.0],
            "seed": [0],
            "test_size": [0.2],
            "max_iter": [1000],
            "lasso": [True],
        }
    ):
        configs.append({**config, "method": "sklearn_saga"})
    return configs
