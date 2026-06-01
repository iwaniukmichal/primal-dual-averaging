from __future__ import annotations

import math

from _shared.grids import parameter_grid
from pda import get_objective


EXPERIMENT = "simplex_entropy_geometry"
DESCRIPTION = "Test entropy prox and L1/L-infinity geometry on simplex objectives."


def build_configs() -> list[dict[str, object]]:
    objectives = ["simplex_linear_8d", "simplex_max_affine_8d",
                  "simplex_sparse_max_affine_16d"]
    configs: list[dict[str, object]] = []
    for objective_id in objectives:
        dimension = get_objective(objective_id).dimension
        for config in parameter_grid(
            {
                "objective_id": [objective_id],
                "D": [0.05, 0.25, 0.75, float(math.log(dimension))],
                "gamma_mult": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
                "dual_averaging": ["simple", "weighted"],
                "restrict_to_fd": [False, True],
                "max_iter": [1500],
                "eps": [1e-4],
            }
        ):
            configs.append(
                {
                    **config,
                    "method": f"sda_{config['dual_averaging']}_entropy",
                    "prox_fun": "entropy",
                }
            )
    return configs
