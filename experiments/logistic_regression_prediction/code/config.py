from __future__ import annotations

from _shared.grids import parameter_grid


EXPERIMENT = "logistic_regression_prediction"
DESCRIPTION = "Compare SDA, weighted dual averaging, projected subgradient, and sklearn on binary logistic regression."


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
        "seed": [0, 1, 2],
        "test_size": [0.2],
        "D": [8.0, 32.0, 128.0],
        "max_iter": [1000],
        "eps": [1e-4],
        "lasso": [False],
    }
    configs: list[dict[str, object]] = []
    for dual_averaging in ["simple", "weighted"]:
        method = f"sda_{dual_averaging}_euclidean"
        for config in parameter_grid({**common, "gamma_mult":  [0.25, 0.5, 1.0, 2.0]}):
            configs.append(
                {
                    **config,
                    "method": method,
                    "prox_fun": "euclidean",
                    "dual_averaging": dual_averaging,
                }
            )
    for config in parameter_grid({**common, "alpha": [0.1, 0.5, 2.0]}):
        configs.append({**config, "method": "projected_subgradient"})
    for config in parameter_grid(
        {
            "dataset": DATASETS,
            "seed": [0, 1, 2],
            "test_size": [0.2],
            "max_iter": [1000],
            "lasso": [False],
        }
    ):
        configs.append({**config, "method": "sklearn_saga"})
    return configs
