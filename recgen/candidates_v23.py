"""v23's neurons and facets: 50 sampled 410m layer-12 neurons per peak quartile whose top 50 form at least
two network-native facets (v6's detector), with those facets (see recgen/rubric_v23.py)."""

import json

import numpy as np

from recgen.analyze_v6 import Layer
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.rubric_v23 import DEPTH, MIN_FACETS, MODEL, quartiles, select
from recgen.sweep import sweep_path

CANDIDATES_PATH = RESULTS_DIR / "candidates_v23.json"
V23_DIR = DATA_DIR / "v23"
RECORDING_PATH = V23_DIR / "step143000.npz"


def peaks(neuron_ids: list[int]) -> np.ndarray:
    """v12's peak: each neuron's largest window maximum in v5's sweep."""
    window_max = np.load(sweep_path(MODEL, DEPTH))["max"]
    return np.array([float(window_max[:, n].astype(np.float32).max()) for n in neuron_ids])


def layer_quartiles() -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Every sampled neuron's id, peak, peak quartile (over all sampled neurons) and v6 facet count."""
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"][f"{MODEL} {DEPTH}"]
    ids = [int(n) for n in v6]
    peak = peaks(ids)
    return ids, peak, quartiles(peak), np.array([v6[str(n)]["n_facets"] for n in ids])


def build_candidates() -> list[dict]:
    """In v13's format (facets as [window, position, 0.0]), plus each neuron's peak, quartile, v6's flag and
    the facet pair v6's reader calls least related."""
    ids, peak, quartile, n_facets = layer_quartiles()
    layer = Layer(MODEL, DEPTH)
    assert [int(n) for n in layer.neurons] == ids, "v6's recording and results should list the same neurons"
    median_peak = float(np.median(peak))
    candidates = []
    for s in select(quartile, n_facets >= MIN_FACETS):
        result = layer.neuron(int(s))
        assert len(result["facets"]) == n_facets[s], "v6's facets should reproduce"
        keys = layer.keys[layer.rows[s]]
        i, j = result["least_related_pair"]
        candidates.append({"neuron": ids[s], "sample": int(s), "quartile": int(quartile[s]), "peak": float(peak[s]),
                           "log_peak": float(np.log(peak[s] / median_peak)), "unrelated_mixed": bool(result["unrelated_mixed"]),
                           "reader_least_pair": f"{i}-{j}",
                           "facets": [[[int(w), int(p), 0.0] for w, p in keys[f]] for f in result["facets"]]})
    return candidates


def load_candidates() -> list[dict]:
    if not CANDIDATES_PATH.exists():
        CANDIDATES_PATH.write_text(json.dumps(build_candidates(), indent=1))
    return json.loads(CANDIDATES_PATH.read_text())
