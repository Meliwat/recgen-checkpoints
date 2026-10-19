"""Apply the pre-registered v5 rules (recgen/rubric_v5.py): v1's facet detector on each
Pythia size and depth.

Reads cached data only: data/v5/*.npz (recgen.sweep). Sentence embeddings of every example
context are computed once and cached in data/v5/embeddings.npz.
"""

import json
from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from recgen.config import RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.facets import centroid, context_text, display_text, embed, find_facets, tokenizer, top_examples
from recgen.rubric import tokens_related
from recgen.rubric_v5 import (
    DEAD_FRACTION, DENSE_MAX_WINDOWS, DENSE_STRONG_FRACTION, DEPTHS, MODELS, PRIMARY_DEPTH, bootstrap_interval, decide,
    eligible, sample, unrelated_mixed,
)
from recgen.sweep import SWEEP_DIR, sweep_path

EMBEDDINGS_PATH = SWEEP_DIR / "embeddings.npz"
V1_MULTI_FACET = (997, 4012)  # v1, 410m layer 12, float32: what `uv run python -m recgen.select` prints
EXAMPLE_NEURONS = 3  # unrelated-mixed neurons shown per model in the report
EMBED_CHUNK = 50_000
# Exploratory, added after review: detector settings (cosine distance, minimum facet size) besides v1's (0.75, 5).
SENSITIVITY_SETTINGS = [(0.7, 5), (0.8, 5), (0.75, 4), (0.75, 6)]
NULL_SHUFFLES = 200


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
            peak = window_max.max(axis=0)
            layers[model, depth] = {
                "layer": int(data["layer"]), "n_layers": int(data["n_layers"]), "width": window_max.shape[1],
                "eligible": len(candidates),
                "near_dead": int((peak < DEAD_FRACTION * np.median(peak)).sum()),
                "dense": int(((window_max > DENSE_STRONG_FRACTION * peak).mean(axis=0) > DENSE_MAX_WINDOWS).sum()),
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


def least_related(embeddings: np.ndarray, facets: list) -> tuple[float | None, tuple[int, int] | None]:
    if len(facets) < 2:
        return None, None
    centroids = [centroid(embeddings[f]) for f in facets]
    return min((float(centroids[i] @ centroids[j]), (i, j)) for i, j in combinations(range(len(facets)), 2))


def facets_with(embeddings: np.ndarray, distance: float, min_size: int) -> list[np.ndarray]:
    """find_facets with other settings, for the exploratory sensitivity check."""
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=distance).fit_predict(embeddings)
    return [c for c in (np.flatnonzero(labels == label) for label in np.unique(labels)) if len(c) >= min_size]


def neuron_facets(examples: list, vectors: dict) -> dict:
    embeddings = np.stack([vectors[w, p] for w, p, _ in examples])
    facets = find_facets(embeddings)
    least, pair = least_related(embeddings, facets)
    similarities = embeddings @ embeddings.T
    off_diagonal = similarities[~np.eye(len(embeddings), dtype=bool)]
    sensitivity = {}
    for distance, min_size in SENSITIVITY_SETTINGS:
        other = facets_with(embeddings, distance, min_size)
        sensitivity[distance, min_size] = (len(other), least_related(embeddings, other)[0])
    return {"n_facets": len(facets), "least_related_similarity": least, "least_related_pair": pair,
            "mean_example_similarity": float(off_diagonal.mean()),
            "share_in_facets": sum(len(f) for f in facets) / len(examples),
            "facets": [[examples[i] for i in f] for f in facets], "sensitivity": sensitivity}


def layer_stats(model: str, depth: str, layer: dict, neurons: dict) -> dict:
    mixed = np.array([unrelated_mixed(r["n_facets"], r["least_related_similarity"]) for r in neurons.values()])
    multi = np.array([r["n_facets"] >= 2 for r in neurons.values()])
    rng = rng_for(model, depth)
    least = [r["least_related_similarity"] for r in neurons.values() if r["least_related_similarity"] is not None]
    return {
        "model": model, "depth": depth, "layer": layer["layer"], "n_layers": layer["n_layers"], "width": layer["width"],
        "eligible": layer["eligible"], "near_dead": layer["near_dead"], "dense": layer["dense"], "sampled": len(neurons),
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


def facet_tokens(facet: list, corpus: np.ndarray) -> tuple[str, str]:
    """A facet's most common activating token and most common next token, normalized."""
    tok = tokenizer()
    current = Counter(tok.decode(corpus[w, p:p + 1]).strip().lower() for w, p, _ in facet)
    following = Counter(tok.decode(corpus[w, p + 1:p + 2]).strip().lower() for w, p, _ in facet if p + 1 < corpus.shape[1])
    return current.most_common(1)[0][0], following.most_common(1)[0][0] if following else ""


def shared_token_rates(neurons: dict, corpus: np.ndarray, rng) -> dict:
    """Exploratory: among unrelated-mixed neurons, how often the least-related pair of facets shares
    its activating token (equal, or one contains the other) or its next token. The null pairs a facet
    of one such neuron with a facet of another, in the same layer."""
    mixed = [r for r in neurons.values() if unrelated_mixed(r["n_facets"], r["least_related_similarity"])]
    pairs = [(facet_tokens(r["facets"][r["least_related_pair"][0]], corpus),
              facet_tokens(r["facets"][r["least_related_pair"][1]], corpus)) for r in mixed]
    if len(pairs) < 2:
        return {"neurons": len(pairs)}

    def rates(ps):  # tokens that strip to nothing (whitespace) never count as shared
        return {"same_activating_token": float(np.mean([a[0] != "" and tokens_related(a[0], b[0]) for a, b in ps])),
                "same_next_token": float(np.mean([a[1] != "" and a[1] == b[1] for a, b in ps]))}

    nulls = []
    for _ in range(NULL_SHUFFLES):
        shuffled = rng.permutation(len(pairs))
        mismatched = [(pairs[i][0], pairs[j][1]) for i, j in enumerate(shuffled) if i != j]
        if mismatched:  # a shuffle of a few neurons can map every facet back to its own neuron
            nulls.append(rates(mismatched))
    actual = rates(pairs)
    return {"neurons": len(pairs), "least_related_pair": {k: round(v, 3) for k, v in actual.items()},
            "null_mean": {k: round(float(np.mean([n[k] for n in nulls])), 4) for k in actual}}


def shares_and_verdicts(flags_by_layer: dict, label: str) -> dict:
    """Share and bootstrap interval of flagged neurons in each layer, and the rubric's verdict at each
    depth, over all six sizes and without 70m."""
    out = {}
    for depth in DEPTHS:
        rows = []
        for model in MODELS:
            flags = np.array(flags_by_layer[model, depth])
            rows.append((float(flags.mean()), bootstrap_interval(flags, rng_for(model, depth))) if len(flags) else (np.nan, (np.nan, np.nan)))
        shares, intervals = [r[0] for r in rows], [r[1] for r in rows]
        all_six = decide(shares, intervals)
        without_70m = decide(shares[1:], intervals[1:])
        out[f"{depth}: {label}"] = {"shares": [round(x, 3) for x in shares],
                                    "verdict": all_six[0], "spearman_rho": round(all_six[1], 3),
                                    "verdict_without_70m": without_70m[0], "spearman_rho_without_70m": round(without_70m[1], 3)}
    return out


def exploratory_stats(stats: list[dict], per_neuron: dict) -> dict:
    """Not pre-registered: checks added after an independent review of the first v5 results."""
    out = {"facet_count_shares": {}, "median_mean_example_similarity": {}, "mean_share_in_facets": {}, "verdicts": {}}
    for s in stats:
        key = f"{s['model']} {s['depth']}"
        counts, total = s["facet_counts"], s["sampled"]
        out["facet_count_shares"][key] = {"0": round(counts.get(0, 0) / total, 3), "1": round(counts.get(1, 0) / total, 3),
                                          "2+": round(sum(v for k, v in counts.items() if k >= 2) / total, 3)}
        rows = per_neuron[s["model"], s["depth"]].values()
        out["median_mean_example_similarity"][key] = round(float(np.median([r["mean_example_similarity"] for r in rows])), 3)
        out["mean_share_in_facets"][key] = round(float(np.mean([r["share_in_facets"] for r in rows])), 3)

    def flags(condition, flag):
        return {key: [flag(r) for r in rows.values() if condition(r)] for key, rows in per_neuron.items()}

    def is_mixed(r):
        return unrelated_mixed(r["n_facets"], r["least_related_similarity"])

    out["verdicts"].update(shares_and_verdicts(flags(lambda r: r["n_facets"] >= 1, is_mixed),
                                               "unrelated mixed among neurons with 1+ facets"))
    out["verdicts"].update(shares_and_verdicts(flags(lambda r: r["n_facets"] >= 2, is_mixed),
                                               "unrelated mixed among neurons with 2+ facets"))
    for setting in SENSITIVITY_SETTINGS:
        out["verdicts"].update(shares_and_verdicts(
            flags(lambda r: True, lambda r, s=setting: unrelated_mixed(*r["sensitivity"][s])),
            f"unrelated mixed, distance {setting[0]}, minimum facet size {setting[1]}"))
    return out


def write_report(stats, per_neuron, corpus) -> None:
    lines = ["# v5 results per model\n",
             f"Share of sampled neurons that are unrelated mixed (2+ facets, least-related similarity below 0.1) and "
             f"that have 2+ facets of any kind, with 95% bootstrap intervals. Then, at each depth, "
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
        for depth in DEPTHS:
            neurons = per_neuron[model, depth]
            mixed = sorted(n for n, r in neurons.items() if unrelated_mixed(r["n_facets"], r["least_related_similarity"]))
            shown = rng_for(model, depth).choice(mixed, size=min(EXAMPLE_NEURONS, len(mixed)), replace=False)
            lines.append(f"### {depth} depth\n")
            for n in sorted(int(x) for x in shown):
                r = neurons[n]
                lines.append(f"**Neuron {n}** ({r['n_facets']} facets, least-related similarity "
                             f"{r['least_related_similarity']:.2f})\n")
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
        "exploratory": exploratory_stats(stats, per_neuron),
        "exploratory_shared_tokens": {f"{model} {depth}": shared_token_rates(per_neuron[model, depth], corpus,
                                                                             rng_for(model, depth))
                                      for model in MODELS for depth in DEPTHS},
    }
    neurons_out = {f"{model} {depth}": {str(n): {k: v for k, v in r.items() if k not in ("facets", "least_related_pair", "sensitivity")}
                                        for n, r in rows.items()}
                   for (model, depth), rows in per_neuron.items()}
    (RESULTS_DIR / "classification_v5.json").write_text(
        json.dumps({"summary": summary, "layers": stats, "neurons": neurons_out}, indent=1))
    write_report(stats, per_neuron, corpus)
    print(json.dumps({"summary": summary, "layers": [{k: s[k] for k in ("model", "depth", "layer", "eligible",
                     "unrelated_mixed_share", "unrelated_mixed_interval", "multi_facet_share")} for s in stats]}, indent=1))


if __name__ == "__main__":
    main()
