"""v13's neurons and facets: network-native facets (v6's detector) of the sampled 410m layer-12 neurons
that the network reader flags unrelated mixed (see recgen/rubric_v13.py)."""

import json

import numpy as np

from recgen.analyze_v6 import Layer
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.rubric_v13 import DEPTH, MODEL, select

CANDIDATES_PATH = RESULTS_DIR / "candidates_v13.json"
V13_DIR = DATA_DIR / "v13"


def v13_path(step: int):
    return V13_DIR / f"step{step}.npz"


def build_candidates() -> list[dict]:
    """In the format recgen.analyze_v4 reads: each neuron's MLP id and its facets as lists of
    [window, position, 0.0] (activations aren't needed downstream)."""
    flags = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"][f"{MODEL} {DEPTH}"]
    layer = Layer(MODEL, DEPTH)
    flagged = np.array([flags[str(int(n))]["unrelated_mixed"] for n in layer.neurons])
    candidates = []
    for s in select(len(layer.neurons), flagged):
        result = layer.neuron(int(s))
        assert result["unrelated_mixed"], "v6's flag should reproduce"
        keys = layer.keys[layer.rows[s]]
        candidates.append({"neuron": int(layer.neurons[s]), "sample": int(s),
                           "facets": [[[int(w), int(p), 0.0] for w, p in keys[f]] for f in result["facets"]]})
    return candidates


def load_candidates() -> list[dict]:
    if not CANDIDATES_PATH.exists():
        CANDIDATES_PATH.write_text(json.dumps(build_candidates(), indent=1))
    return json.loads(CANDIDATES_PATH.read_text())
