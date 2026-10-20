"""v7 units: real neurons, pooled pairs of random neurons, and pooled pairs of related neurons,
at the middle layer of each size, built from v6's recordings (the same sampled neurons and examples)."""

import numpy as np

from recgen.rubric_v5 import MODELS
from recgen.rubric_v7 import DEPTH, MIXES, POOL_SIZE, RETEST, pool_rows, random_pairs, real_units, related_pairs, rng_for
from recgen.sweep_v6 import v6_path


def build_units(model: str) -> list[dict]:
    """Every unit of one model: id, kind, mix, the sampled neurons it pools, and its example rows
    (into v6's recording) and positions (window, position)."""
    data = np.load(v6_path(model, DEPTH))
    keys, rows = data["keys"], data["example_rows"]
    directions = data["weights"].astype(np.float64) * data["gain"].astype(np.float64)
    units = []

    def add(kind: str, mix: str | None, neurons: tuple[int, ...], unit_rows: np.ndarray) -> None:
        units.append({"id": f"{model}:{kind}:{mix or ''}:{'-'.join(map(str, neurons))}", "model": model, "kind": kind,
                      "mix": mix, "neurons": neurons, "rows": unit_rows, "keys": keys[unit_rows]})

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
    """The units of one model that the language model reads a second time."""
    per_model = RETEST // len(MODELS)
    return [units[i]["id"] for i in sorted(rng_for(model, 9).choice(len(units), size=per_model, replace=False))]
