"""Apply the pre-registered rubric to the recorded trajectories."""

import json
from collections import Counter
from itertools import combinations

import numpy as np

from recgen.activations import MIN_CONTEXT
from recgen.config import CHECKPOINT_STEPS, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import TOP_K, centroid, context_text, embed, tokenizer
from recgen.record import neuron_activations_path
from recgen.rubric import Facet, classify, emergence

ADJACENT_PERCENTILE = 75
# Neurons that look mixed to a human reader, with a note on why, chosen from the
# candidates before any trajectory was recorded.
HAND_PICKED_PATH = RESULTS_DIR / "hand_picked.json"
TOP_SHOWN = 10  # strongest examples kept per checkpoint for reading trajectories


def activating_token(corpus, window: int, pos: int) -> str:
    return tokenizer().decode(corpus[window, pos:pos + 1]).strip().lower()


def window_top(acts: np.ndarray, k: int):
    """Strongest k windows of one neuron's activations [W, T]: (window, pos, act) strongest first."""
    maxima = acts.max(axis=1)
    windows = np.argsort(-maxima)[:k]
    return [(int(w), int(acts[w].argmax()), float(maxima[w])) for w in windows]


def trace_neuron(acts: np.ndarray, facets: list[list]) -> dict:
    """One checkpoint, one neuron: each facet's response and the strongest examples."""
    top = window_top(acts, TOP_K)
    reference = float(np.median([a for _, _, a in top]))
    responses = [float(np.median([acts[w, p] for w, p, _ in facet])) / reference for facet in facets]
    return {"reference": reference, "responses": responses, "top": top[:TOP_SHOWN]}


def trace(candidates: list[dict]) -> dict:
    trajectories = {c["neuron"]: {} for c in candidates}
    for step in CHECKPOINT_STEPS:
        data = np.load(neuron_activations_path(step))
        column = {int(n): i for i, n in enumerate(data["neurons"])}
        acts = data["acts"].astype(np.float32)
        acts[:, :MIN_CONTEXT] = -np.inf
        for c in candidates:
            trajectories[c["neuron"]][step] = trace_neuron(acts[:, :, column[c["neuron"]]], c["facets"])
    return trajectories


def build_facets(candidates, trajectories, corpus) -> dict[int, list[Facet]]:
    facets = {}
    for c in candidates:
        n = c["neuron"]
        responses = np.array([trajectories[n][s]["responses"] for s in CHECKPOINT_STEPS]).T
        facets[n] = []
        for facet_examples, facet_responses in zip(c["facets"], responses):
            embeddings = embed([context_text(corpus, w, p) for w, p, _ in facet_examples])
            token = Counter(activating_token(corpus, w, p) for w, p, _ in facet_examples).most_common(1)[0][0]
            facets[n].append(Facet(facet_responses, centroid(embeddings), token))
    return facets


def adjacency_threshold(facets: dict[int, list[Facet]]) -> float:
    """Similarity above which two facets count as adjacent: the given percentile of
    similarities between facets of different neurons."""
    similarities = [float(f.centroid @ g.centroid)
                    for a, b in combinations(facets, 2) for f in facets[a] for g in facets[b]]
    return float(np.percentile(similarities, ADJACENT_PERCENTILE))


def main() -> None:
    corpus = load_corpus()
    candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    hand_picked = json.loads(HAND_PICKED_PATH.read_text())
    trajectories = trace(candidates)
    facets = build_facets(candidates, trajectories, corpus)
    threshold = adjacency_threshold(facets)

    results = []
    for c in candidates:
        n = c["neuron"]
        results.append({
            "neuron": n,
            "hand_picked": str(n) in hand_picked,
            "label": classify(facets[n], threshold),
            "emergence_steps": [None if (i := emergence(f.responses)) is None else CHECKPOINT_STEPS[i]
                                for f in facets[n]],
            "top_tokens": [f.top_token for f in facets[n]],
            "facet_similarities": {f"{i}-{j}": round(float(facets[n][i].centroid @ facets[n][j].centroid), 3)
                                   for i, j in combinations(range(len(facets[n])), 2)},
        })
    summary = {
        "adjacency_threshold": round(threshold, 3),
        "labels_all": Counter(r["label"] for r in results),
        "labels_hand_picked": Counter(r["label"] for r in results if r["hand_picked"]),
    }
    (RESULTS_DIR / "classification.json").write_text(json.dumps({"summary": summary, "neurons": results}, indent=1))
    (RESULTS_DIR / "trajectories.json").write_text(json.dumps(trajectories))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
