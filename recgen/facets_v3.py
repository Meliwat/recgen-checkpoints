"""Enlarge v1's facets with more of each neuron's strong windows (rule in recgen/rubric_v3.py).

Input side only: uses the final checkpoint's window maxima and the sentence model, as v1 did.
Writes results/facets_v3.json: for each traced neuron, each facet's examples as
[window, position, activation, similarity to the facet's v1 centroid, whether v1 had it].
"""

import json

import numpy as np

from recgen.config import CONTEXT_LEN, FINAL_STEP, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import TOP_K, centroid, context_text, embed, top_examples
from recgen.record import window_maxima_path
from recgen.rubric_v3 import POOL_FRACTION, POOL_MAX, enlarge, scorable

FACETS_V3_PATH = RESULTS_DIR / "facets_v3.json"


def pool(window_max: np.ndarray, window_pos: np.ndarray, neuron: int):
    """The neuron's strongest windows down to POOL_FRACTION of its top-50 median level."""
    examples = top_examples(window_max, window_pos, neuron, k=POOL_MAX)
    level = np.median([a for _, _, a in examples[:TOP_K]])
    return [e for e in examples if e[2] >= POOL_FRACTION * level]


def enlarged_facets(candidate: dict, corpus: np.ndarray, window_max: np.ndarray, window_pos: np.ndarray):
    examples = pool(window_max, window_pos, candidate["neuron"])
    row = {(w, p): i for i, (w, p, _) in enumerate(examples)}
    embeddings = embed([context_text(corpus, w, p) for w, p, _ in examples])
    originals = [[row[(w, p)] for w, p, _ in facet] for facet in candidate["facets"]]
    centroids = np.array([centroid(embeddings[rows]) for rows in originals])
    similarities = embeddings @ centroids.T
    keep = np.array([scorable(p, CONTEXT_LEN) for _, p, _ in examples])
    original_rows = {i for rows in originals for i in rows}
    return [[[examples[i][0], examples[i][1], round(examples[i][2], 4), round(float(similarities[i, f]), 4),
              i in original_rows] for i in rows]
            for f, rows in enumerate(enlarge(similarities, originals, keep))]


def main() -> None:
    if FACETS_V3_PATH.exists():
        print(f"{FACETS_V3_PATH.name}: already built")
        return
    corpus = load_corpus()
    candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    data = np.load(window_maxima_path(FINAL_STEP))
    result = [{"neuron": c["neuron"], "facets": enlarged_facets(c, corpus, data["max"], data["pos"])}
              for c in candidates]
    FACETS_V3_PATH.write_text(json.dumps(result))
    sizes = [len(f) for r in result for f in r["facets"]]
    originals = [sum(e[4] for e in f) for r in result for f in r["facets"]]
    print(f"{len(sizes)} facets: {sum(originals)} v1 examples kept, {sum(sizes)} in all; "
          f"sizes {min(sizes)} to {max(sizes)}, median {int(np.median(sizes))}")


if __name__ == "__main__":
    main()
