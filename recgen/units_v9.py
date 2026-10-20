"""v9 units: unions of two real neurons at the middle layer, built from v5's window maxima and v6's
recorded examples (see recgen/rubric_v9.py)."""

import numpy as np

from recgen.rubric_v7 import DEPTH
from recgen.rubric_v9 import N_UNION, PRIMARY, minority_count, rng_for, scale, union_top
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path

MAX_DRAWS = 10 * N_UNION


def build_unions(model: str, how: str = PRIMARY, pairs: list[tuple[int, int]] | None = None) -> tuple[list[dict], int]:
    """Union units for one model, and how many drawn pairs were replaced. With pairs given (for the
    sensitivity check), uses exactly those pairs and drops any that fail instead of replacing them."""
    v6 = np.load(v6_path(model, DEPTH))
    v5 = np.load(sweep_path(model, DEPTH))
    window_max, window_pos = v5["max"], v5["pos"]
    neurons, example_rows, keys = v6["neurons"], v6["example_rows"], v6["keys"]
    row_of = {(int(w), int(p)): i for i, (w, p) in enumerate(keys)}
    scales = {}

    def scale_of(s: int) -> float:
        if s not in scales:
            scales[s] = scale(window_max[:, neurons[s]].astype(np.float32), how)
        return scales[s]

    def unit(a: int, b: int) -> dict | None:
        top = union_top(window_max[:, neurons[a]], window_pos[:, neurons[a]],
                        window_max[:, neurons[b]], window_pos[:, neurons[b]], scale_of(a), scale_of(b))
        own = set(example_rows[a]) | set(example_rows[b])
        rows = [row_of.get((w, p)) for w, p, _ in top]
        if any(r is None or r not in own for r in rows):
            return None
        sources = [source for _, _, source in top]
        return {"id": f"{model}:union:{how}:{a}-{b}", "model": model, "kind": "union", "how": how, "neurons": (a, b),
                "neuron_ids": [int(neurons[a]), int(neurons[b])], "rows": np.array(rows), "keys": keys[np.array(rows)],
                "sources": sources, "minority": minority_count(sources)}

    units, replaced = [], 0
    if pairs is not None:
        built = [unit(a, b) for a, b in pairs]
        return [u for u in built if u is not None], sum(u is None for u in built)
    rng = rng_for(model, 0)
    for _ in range(MAX_DRAWS):
        a, b = (int(x) for x in rng.choice(len(neurons), size=2, replace=False))
        u = unit(a, b)
        if u is None:
            replaced += 1
            continue
        units.append(u)
        if len(units) == N_UNION:
            break
    assert len(units) == N_UNION, f"{model}: only {len(units)} unions in {MAX_DRAWS} draws"
    return units, replaced
