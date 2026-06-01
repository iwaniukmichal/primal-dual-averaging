from __future__ import annotations

from itertools import product
from typing import Any, Iterable


def parameter_grid(options: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Return a deterministic Cartesian product of experiment parameters."""
    keys = list(options)
    values = [list(options[key]) for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def with_limit(configs: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Apply an optional positive run limit."""
    if limit is None:
        return configs
    return configs[: max(0, int(limit))]

