from __future__ import annotations

from _shared.grids import parameter_grid


EXPERIMENT = "lasso_logistic_sparsity_ssa_comparison"
DESCRIPTION = "Compare weighted SDA and stochastic simple averages on lasso logistic datasets."


DATASETS = [
    "synthetic_logistic_small_5d.csv",
    "synthetic_logistic_sparse_20d.csv",
    "synthetic_logistic_noisy_20d.csv",
    "synthetic_logistic_imbalanced_10d.csv",
    "synthetic_logistic_big_sparse_50d.csv",
    "synthetic_logistic_big_noisy_80d.csv",
]


def build_configs() -> list[dict[str, object]]:
    common = {
        "dataset": DATASETS,
        "lambda": [1.0],
        "restrict_to_fd": [False],
        "seed": [0],
        "test_size": [0.2],
        "D": [128.0],
        "max_iter": [1000],
        "eps": [1e-3],
        "lasso": [True],
    }
    configs: list[dict[str, object]] = []
    for config in parameter_grid({**common, "gamma_mult": [0.25, 0.5, 1.0, 2.0]}):
        configs.append(
            {
                **config,
                "method": "sda_weighted_euclidean",
                "prox_fun": "euclidean",
                "dual_averaging": "weighted",
            }
        )
    for config in parameter_grid({**common, "gamma_mult": [0.25, 0.5, 1.0, 2.0]}):
        configs.append(
            {
                **config,
                "method": "ssa_euclidean",
                "prox_fun": "euclidean",
                "dual_averaging": "stochastic_simple",
                "batch_size": 1,
            }
        )
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
