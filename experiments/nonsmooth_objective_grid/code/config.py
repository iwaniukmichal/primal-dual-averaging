from __future__ import annotations

from _shared.grids import parameter_grid
from _shared.objectives import registry_objective_ids


EXPERIMENT = "nonsmooth_objective_grid"
DESCRIPTION = "Compare SDA, weighted dual averaging, and projected subgradient on nonsmooth objectives."


def build_configs() -> list[dict[str, object]]:
    objectives = registry_objective_ids(include_simplex=False)
    common = {
        "objective_id": objectives,
        "D": [0.5, 2.0, 8.0, 32.0],
        "restrict_to_fd": [False, True],
        "max_iter": [1000],
        "eps": [1e-4],
    }
    configs: list[dict[str, object]] = []
    for method, dual_averaging in [
        ("sda_simple_euclidean", "simple"),
        ("sda_weighted_euclidean_dual_average", "weighted"),
    ]:
        for config in parameter_grid({**common, "gamma_mult": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]}):
            configs.append(
                {
                    **config,
                    "method": method,
                    "prox_fun": "euclidean",
                    "dual_averaging": dual_averaging,
                }
            )
    for config in parameter_grid({**common, "alpha": [0.1, 0.25, 0.5, 1.0, 2.0]}):
        configs.append({**config, "method": "projected_subgradient"})
    return configs
