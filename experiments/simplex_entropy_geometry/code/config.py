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
        for config in parameter_grid(
            {
                "objective_id": [objective_id],
                "D": [1, 2, 3],
                "gamma_mult": [0.1, 0.5, 1.0, 4.0],
                "dual_averaging": ["simple", "weighted"],
                "restrict_to_fd": [False],
                "max_iter": [1000],
                "eps": [1e-3],
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
