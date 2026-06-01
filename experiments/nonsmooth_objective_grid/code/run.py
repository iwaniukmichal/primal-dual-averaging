from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _shared.bootstrap import PROJECT_ROOT  # noqa: F401
from _shared.experiment_main import run_experiment_main
from config import DESCRIPTION, EXPERIMENT, build_configs


if __name__ == "__main__":
    run_experiment_main(
        experiment=EXPERIMENT,
        description=DESCRIPTION,
        configs=build_configs(),
        runner_kind="registry",
        run_file=__file__,
    )

