"""Pick neurons whose strongest examples at the final checkpoint look mixed:
two or more facets that an independent sentence model sees as unrelated.

Uses only the final checkpoint, so the choice can't be influenced by trajectories.
"""

import json
from itertools import combinations

import numpy as np

from recgen.config import FINAL_STEP, LAYER, MODEL, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import centroid, context_text, display_text, embed, find_facets, top_examples
from recgen.record import SELECTED_PATH, window_maxima_path

MIN_PEAK = 1.0  # skip near-dead neurons (typical peak activation here is ~2)
STRONG_FRACTION = 0.5  # a window counts as strong if it reaches half the neuron's peak
MAX_STRONG_WINDOWS = 0.5  # skip dense neurons; most reach 30% of peak somewhere in 80% of 512-token windows
N_TRACED = 40
EXAMPLES_SHOWN = 4


def facet_summary(neuron: int, corpus, window_max, window_pos) -> dict | None:
    examples = top_examples(window_max, window_pos, neuron)
    embeddings = embed([context_text(corpus, w, p) for w, p, _ in examples])
    facets = find_facets(embeddings)
    if len(facets) < 2:
        return None
    centroids = [centroid(embeddings[f]) for f in facets]
    least_related = min(float(a @ b) for a, b in combinations(centroids, 2))
    return {
        "neuron": neuron,
        "least_related_similarity": round(least_related, 3),
        "facets": [[examples[i] for i in f] for f in facets],
    }


def main() -> None:
    corpus = load_corpus()
    data = np.load(window_maxima_path(FINAL_STEP))
    window_max, window_pos = data["max"], data["pos"]
    peak = window_max.max(axis=0)
    strong_share = (window_max > STRONG_FRACTION * peak).mean(axis=0)
    eligible = np.flatnonzero((peak > MIN_PEAK) & (strong_share < MAX_STRONG_WINDOWS))
    print(f"{len(eligible)} of {window_max.shape[1]} neurons are neither near-dead nor dense")

    summaries = [s for n in eligible if (s := facet_summary(int(n), corpus, window_max, window_pos))]
    summaries.sort(key=lambda s: s["least_related_similarity"])
    print(f"{len(summaries)} have two or more facets")

    candidates = summaries[:N_TRACED]
    (RESULTS_DIR / "candidates.json").write_text(json.dumps(candidates, indent=1))
    lines = [f"# Candidate neurons, layer {LAYER} of {MODEL} at step {FINAL_STEP}\n",
             f"Ranked by how unrelated their two least-related facets are (centroid cosine similarity).\n"]
    for rank, s in enumerate(candidates, 1):
        lines.append(f"## {rank}. Neuron {s['neuron']} (least-related facet similarity {s['least_related_similarity']})\n")
        for i, facet in enumerate(s["facets"]):
            lines.append(f"**Facet {i}** ({len(facet)} examples)\n")
            lines += [f"- {a:.2f} `{display_text(corpus, w, p)}`" for w, p, a in facet[:EXAMPLES_SHOWN]]
            lines.append("")
    (RESULTS_DIR / "candidates.md").write_text("\n".join(lines))
    SELECTED_PATH.write_text(json.dumps({"traced": [s["neuron"] for s in candidates]}, indent=1))


if __name__ == "__main__":
    main()
