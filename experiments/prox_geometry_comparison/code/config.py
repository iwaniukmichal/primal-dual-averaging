from __future__ import annotations

from _shared.grids import parameter_grid


EXPERIMENT = "prox_geometry_comparison"
DESCRIPTION = "Compare Euclidean and weighted Euclidean prox choices on coordinate-imbalanced objectives."


def build_configs() -> list[dict[str, object]]:
    common = {
        "objective_id": [
            "ill_conditioned_l1_shift_8d",
            "ill_conditioned_max_affine_8d",
            "weighted_l1_shift_4d",
        ],
        "D": [2.0, 8.0, 32.0],
        "gamma_mult": [0.5, 1.0, 2.0],
        "restrict_to_fd": [False, True],
        "max_iter": [1500],
        "eps": [1e-4],
    }
    configs: list[dict[str, object]] = []
    for prox_fun in ["euclidean", "weighted_euclidean"]:
        profiles = ["none"] if prox_fun == "euclidean" else [
            "uniform",
            "coordinate_scale",
            "inverse_coordinate_scale",
        ]
        for dual_averaging in ["simple", "weighted"]:
            method = f"sda_{dual_averaging}_{prox_fun}"
            for config in parameter_grid({**common, "prox_weights_profile": profiles}):
                configs.append(
                    {
                        **config,
                        "method": method,
                        "prox_fun": prox_fun,
                        "dual_averaging": dual_averaging,
                    }
                )
    return configs

