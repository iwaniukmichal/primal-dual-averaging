from __future__ import annotations

from _shared.grids import parameter_grid


EXPERIMENT = "stopping_and_fd_ablation"
DESCRIPTION = "Isolate D, stopping tolerance, and F_D restriction effects for SDA variants."


def build_configs() -> list[dict[str, object]]:
    common = {
        "D": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
        "eps": [1e-2, 1e-3, 1e-4, 0.0],
        "restrict_to_fd": [False, True],
        "gamma_mult": [0.1, 0.5, 1.0, 2.0],
        "max_iter": [2000],
    }
    configs: list[dict[str, object]] = []
    for objective_id in ["max_affine_4d", "weighted_l1_shift_4d"]:
        for dual_averaging in ["simple", "weighted"]:
            for config in parameter_grid({**common, "objective_id": [objective_id]}):
                configs.append(
                    {
                        **config,
                        "method": f"sda_{dual_averaging}_euclidean",
                        "prox_fun": "euclidean",
                        "dual_averaging": dual_averaging,
                        "runner_kind": "registry",
                    }
                )
    for dual_averaging in ["simple", "weighted"]:
        for config in parameter_grid(
            {
                **common,
                "dataset": ["synthetic_logistic_small_5d.csv"],
                "seed": [0],
                "test_size": [0.2],
            }
        ):
            configs.append(
                {
                    **config,
                    "method": f"sda_{dual_averaging}_euclidean",
                    "prox_fun": "euclidean",
                    "dual_averaging": dual_averaging,
                    "runner_kind": "logistic",
                    "lasso": False,
                }
            )
    return configs
