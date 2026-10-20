"""v19 units: v7's real neurons, pooled pairs of random neurons and pooled pairs of related neurons, at the
middle layer of each GPT-2 size, built from v19's recordings with v19's seeds."""

import numpy as np

from recgen.gpt2_v19 import network_path
from recgen.rubric_v7 import MIXES, POOL_SIZE, pool_rows, related_pairs
from recgen.rubric_v19 import random_pairs, real_units, retest_indices


def build_units(model: str, path=None) -> list[dict]:
    """Every unit of one model: id, kind, mix, the sampled neurons it pools (indices into the sample), and its
    example rows (into the recording) and positions (window, position). path: a recording in v19's format elsewhere
    (v22's, of the same neurons on another corpus)."""
    data = np.load(path or network_path(model))
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
    """The units of one model that Haiku reads a second time."""
    return [units[i]["id"] for i in retest_indices(len(units), model)]
