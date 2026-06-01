from __future__ import annotations

from _shared.grids import parameter_grid


EXPERIMENT = "runtime_scaling"
DESCRIPTION = "Measure runtime and predictive quality as synthetic logistic data size and dimension grow."


def build_configs() -> list[dict[str, object]]:
    common = {
        "n_samples": [200, 1000, 5000],
        "dimension": [5, 20, 100],
        "seed": [0, 1, 2],
        "flip_prob": [0.0, 0.1],
        "test_size": [0.2],
        "D": [32.0],
        "gamma_mult": [1.0],
        "alpha": [0.5],
        "max_iter": [1000],
        "eps": [1e-4],
        "lasso": [False],
    }
    configs: list[dict[str, object]] = []
    for method, dual_averaging in [
        ("sda_simple_euclidean", "simple"),
        ("sda_weighted_euclidean", "weighted"),
        ("projected_subgradient", ""),
        ("sklearn_saga", ""),
    ]:
        for config in parameter_grid(common):
            row = {**config, "method": method}
            if dual_averaging:
                row.update({"prox_fun": "euclidean", "dual_averaging": dual_averaging})
            configs.append(row)
    return configs

