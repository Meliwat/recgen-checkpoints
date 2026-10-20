"""v11 units: v7's real, pooled and related units (recgen/units_v7.py) at any depth, built from v6's
recordings. The same seeded draws are used at every depth, so the middle depth rebuilds v7's units."""

import numpy as np

from recgen.rubric_v7 import MIXES, POOL_SIZE, pool_rows, random_pairs, real_units, related_pairs
from recgen.rubric_v7 import rng_for as v7_rng_for
from recgen.rubric_v11 import RETEST_PER_MODEL, depth_key
from recgen.sweep_v6 import v6_path


def build_units(model: str, depth: str) -> list[dict]:
    """Every unit of one model and depth: id, kind, mix, the sampled neurons it pools, and its example
    rows (into v6's recording) and positions (window, position)."""
    data = np.load(v6_path(model, depth))
    keys, rows = data["keys"], data["example_rows"]
    directions = data["weights"].astype(np.float64) * data["gain"].astype(np.float64)
    units = []

    def add(kind: str, mix: str | None, neurons: tuple[int, ...], unit_rows: np.ndarray) -> None:
        units.append({"id": f"{model}:{depth_key(depth)}:{kind}:{mix or ''}:{'-'.join(map(str, neurons))}",
                      "model": model, "depth": depth, "kind": kind, "mix": mix, "neurons": neurons,
                      "rows": unit_rows, "keys": keys[unit_rows]})

    for s in real_units(len(rows), model):
        add("real", None, (int(s),), rows[s])
    for mix, from_a in MIXES.items():
        for a, b in random_pairs(len(rows), model, mix):
            add("pooled", mix, (a, b), pool_rows(rows[a], rows[b], from_a))
    for a, b in related_pairs(directions):
        add("related", "50/50", (a, b), pool_rows(rows[a], rows[b], MIXES["50/50"]))
    assert all(len(u["rows"]) == POOL_SIZE for u in units)
    return units


def retest_ids(units: list[dict], model: str) -> list[str]:
    """The units of one model and depth that the language model reads a second time."""
    chosen = v7_rng_for(model, 9).choice(len(units), size=RETEST_PER_MODEL, replace=False)
    return [units[i]["id"] for i in sorted(chosen)]

