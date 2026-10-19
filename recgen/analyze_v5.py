"""Apply the pre-registered v5 rules (recgen/rubric_v5.py): v1's facet detector on each
Pythia size and depth.

Reads cached data only: data/v5/*.npz (recgen.sweep). Sentence embeddings of every example
context are computed once and cached in data/v5/embeddings.npz.
"""

import json
from collections import Counter
from itertools import combinations

import numpy as np

from recgen.config import RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.facets import centroid, context_text, display_text, embed, find_facets, top_examples
from recgen.rubric_v5 import (
    DEPTHS, MODELS, PRIMARY_DEPTH, bootstrap_interval, decide, eligible, sample, unrelated_mixed,
)
from recgen.sweep import SWEEP_DIR, sweep_path

EMBEDDINGS_PATH = SWEEP_DIR / "embeddings.npz"
V1_MULTI_FACET = (997, 4012)  # v1, 410m layer 12, float32, v1's eligibility rule
EXAMPLE_NEURONS = 3  # unrelated-mixed neurons shown per model in the report
EMBED_CHUNK = 50_000


def rng_for(model: str, depth: str):
    return np.random.default_rng([SEED, MODELS.index(model), list(DEPTHS).index(depth)])


def sampled_examples() -> dict[tuple[str, str], dict]:
    """For each model and depth: the layer, eligible count, and each sampled neuron's top examples."""
    layers = {}
    for model in MODELS:
        for depth in DEPTHS:
            data = np.load(sweep_path(model, depth))
            window_max, window_pos = data["max"].astype(np.float32), data["pos"]
            candidates = eligible(window_max)
            neurons = sample(candidates, rng_for(model, depth))
            layers[model, depth] = {
                "layer": int(data["layer"]), "n_layers": int(data["n_layers"]), "width": window_max.shape[1],
                "eligible": len(candidates),
                "examples": {int(n): top_examples(window_max, window_pos, int(n)) for n in neurons},
            }
    return layers


def embeddings_for(layers: dict, corpus: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    keys = sorted({(w, p) for layer in layers.values() for examples in layer["examples"].values() for w, p, _ in examples})
    if EMBEDDINGS_PATH.exists():
        cached = np.load(EMBEDDINGS_PATH)
        if [tuple(k) for k in cached["keys"]] == keys:
            return dict(zip(keys, cached["vectors"].astype(np.float32)))
    vectors = np.concatenate([embed([context_text(corpus, w, p) for w, p in keys[i:i + EMBED_CHUNK]])
                              for i in range(0, len(keys), EMBED_CHUNK)])
    np.savez(EMBEDDINGS_PATH, keys=np.array(keys), vectors=vectors.astype(np.float16))
    return dict(zip(keys, vectors))


def neuron_facets(examples: list, vectors: dict) -> dict:
    embeddings = np.stack([vectors[w, p] for w, p, _ in examples])
    facets = find_facets(embeddings)
    least = None
    if len(facets) >= 2:
        centroids = [centroid(embeddings[f]) for f in facets]
        least = min(float(a @ b) for a, b in combinations(centroids, 2))
    return {"n_facets": len(facets), "least_related_similarity": least,
            "facets": [[examples[i] for i in f] for f in facets]}


def layer_stats(model: str, depth: str, layer: dict, neurons: dict) -> dict:
    mixed = np.array([unrelated_mixed(r["n_facets"], r["least_related_similarity"]) for r in neurons.values()])
    multi = np.array([r["n_facets"] >= 2 for r in neurons.values()])
    rng = rng_for(model, depth)
    least = [r["least_related_similarity"] for r in neurons.values() if r["least_related_similarity"] is not None]
    return {
        "model": model, "depth": depth, "layer": layer["layer"], "n_layers": layer["n_layers"], "width": layer["width"],
        "eligible": layer["eligible"], "sampled": len(neurons),
        "unrelated_mixed_share": float(mixed.mean()), "unrelated_mixed_interval": bootstrap_interval(mixed, rng),
        "multi_facet_share": float(multi.mean()), "multi_facet_interval": bootstrap_interval(multi, rng),
        "facet_counts": Counter(r["n_facets"] for r in neurons.values()),
        "median_least_related_similarity": float(np.median(least)) if least else None,
    }


def decisions(stats: list[dict]) -> dict:
    out = {}
    for depth in DEPTHS:
        rows = [next(s for s in stats if s["model"] == m and s["depth"] == depth) for m in MODELS]
        for measure in ("unrelated_mixed", "multi_facet"):
            verdict, rho = decide([r[f"{measure}_share"] for r in rows], [r[f"{measure}_interval"] for r in rows])
            out[f"{depth}: {measure}"] = {"verdict": verdict, "spearman_rho": round(rho, 3),
                                          "primary": depth == PRIMARY_DEPTH and measure == "unrelated_mixed"}
    return out


def write_report(stats, per_neuron, corpus) -> None:
    lines = ["# v5 results per model\n",
             f"Share of sampled neurons that are unrelated mixed (2+ facets, least-related similarity below 0.1) and "
             f"that have 2+ facets of any kind, with 95% bootstrap intervals. Then, for the {PRIMARY_DEPTH} layer, "
             f"{EXAMPLE_NEURONS} randomly chosen unrelated-mixed neurons with two examples per facet.\n"]
    for model in MODELS:
        lines.append(f"## pythia-{model}\n")
        lines.append("| Depth | Layer | Eligible | Sampled | Unrelated mixed | 2+ facets |\n|---|---|---|---|---|---|")
        for s in (s for s in stats if s["model"] == model):
            lo, hi = s["unrelated_mixed_interval"]
            mlo, mhi = s["multi_facet_interval"]
            lines.append(f"| {s['depth']} | {s['layer']} of {s['n_layers']} | {s['eligible']} | {s['sampled']} | "
                         f"{s['unrelated_mixed_share']:.3f} ({lo:.3f} to {hi:.3f}) | "
                         f"{s['multi_facet_share']:.3f} ({mlo:.3f} to {mhi:.3f}) |")
        lines.append("")
        neurons = per_neuron[model, PRIMARY_DEPTH]
        mixed = sorted(n for n, r in neurons.items() if unrelated_mixed(r["n_facets"], r["least_related_similarity"]))
        shown = rng_for(model, PRIMARY_DEPTH).choice(mixed, size=min(EXAMPLE_NEURONS, len(mixed)), replace=False)
        for n in sorted(int(x) for x in shown):
            r = neurons[n]
            lines.append(f"**Neuron {n}** ({r['n_facets']} facets, least-related similarity {r['least_related_similarity']:.2f})\n")
            for i, facet in enumerate(r["facets"]):
                lines += [f"- facet {i}: {a:.2f} `{display_text(corpus, w, p)}`" for w, p, a in facet[:2]]
            lines.append("")
    (RESULTS_DIR / "v5_models.md").write_text("\n".join(lines))


def main() -> None:
    corpus = load_corpus()
    layers = sampled_examples()
    vectors = embeddings_for(layers, corpus)
    per_neuron = {key: {n: neuron_facets(examples, vectors) for n, examples in layer["examples"].items()}
                  for key, layer in layers.items()}
    stats = [layer_stats(model, depth, layers[model, depth], per_neuron[model, depth])
             for model in MODELS for depth in DEPTHS]
    v1_410m = next(s for s in stats if s["model"] == "410m" and s["depth"] == "middle")
    summary = {
        "decisions": decisions(stats),
        "consistency_410m_middle_vs_v1": {"v5_multi_facet_share": round(v1_410m["multi_facet_share"], 3),
                                          "v1_multi_facet_share": round(V1_MULTI_FACET[0] / V1_MULTI_FACET[1], 3)},
        "unique_contexts_embedded": len(vectors),
    }
    neurons_out = {f"{model} {depth}": {str(n): {k: v for k, v in r.items() if k != "facets"} for n, r in rows.items()}
                   for (model, depth), rows in per_neuron.items()}
    (RESULTS_DIR / "classification_v5.json").write_text(
        json.dumps({"summary": summary, "layers": stats, "neurons": neurons_out}, indent=1))
    write_report(stats, per_neuron, corpus)
    print(json.dumps({"summary": summary, "layers": [{k: s[k] for k in ("model", "depth", "layer", "eligible",
                     "unrelated_mixed_share", "unrelated_mixed_interval", "multi_facet_share")} for s in stats]}, indent=1))


if __name__ == "__main__":
    main()
