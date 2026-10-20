"""Apply the pre-registered v13 rules (recgen/rubric_v13.py): v4's formation-time test on network-native
facets of neurons the network reader calls mixed. Reads data/v13/ (recgen.record_v13) and v4's results.
"""

import json
from collections import Counter
from itertools import combinations

import numpy as np

from recgen.analyze_v4 import neuron_results, step_summary
from recgen.analyze_v6 import PRIMARY, Layer
from recgen.candidates_v13 import load_candidates, v13_path
from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.rubric_v2 import null_threshold, overlap
from recgen.rubric_v4 import STEPS, TOP_UNITS, formation_window
from recgen.rubric_v13 import DEPTH, MODEL


def final_adjacency(final: dict, candidates: list[dict], n: int) -> tuple[dict, dict, float]:
    """v2's rule at the final checkpoint: each pair's MLP driver overlap, whether it beats the 95th
    percentile of this neuron's null (its facets against the other neurons' facets), and that threshold."""
    own = [key for key in final["mlp"] if key[0] == n and key[1][0] == n]
    keys = [(c["neuron"], f) for c in candidates for f in range(len(c["facets"]))]
    null = [overlap(final["mlp"][n, a], final["mlp"][n, x], TOP_UNITS) for a in (k[1] for k in own) for x in keys if x[0] != n]
    threshold = null_threshold(null)
    overlaps, adjacent = {}, {}
    for (_, a), (_, b) in combinations(sorted(own), 2):
        key = f"{a[1]}-{b[1]}"
        overlaps[key] = overlap(final["mlp"][n, a], final["mlp"][n, b], TOP_UNITS)
        adjacent[key] = {"v2": bool(overlaps[key] > threshold)}
    return overlaps, adjacent, float(threshold)


def chance_tail(rates: list[float], observed: int) -> float:
    """Exploratory, added after seeing the counts: the probability of at least `observed` branched-then-separated
    pairs if each unrelated pair crossed its threshold only at its own null's false-positive rate, independently.
    Pairs within a neuron share facets, so this is approximate."""
    distribution = np.array([1.0])
    for rate in rates:
        distribution = np.convolve(distribution, [1 - rate, rate])
    return float(distribution[observed:].sum())


def pair_tokens(candidates: list[dict], corpus: np.ndarray) -> dict:
    """Each facet's activating tokens, by (neuron, facet)."""
    return {(c["neuron"], f): [int(corpus[w, p]) for w, p, _ in facet] for c in candidates for f, facet in enumerate(c["facets"])}


def chance_split(pairs: list[dict]) -> dict:
    rates = [p["null_exceedances"] / p["null_size"] for p in pairs]
    branched = sum(p["branched_then_separated"] for p in pairs)
    return {"unrelated_pairs": len(pairs), "branched_then_separated": branched,
            "expected_by_chance": round(sum(rates), 1), "chance_tail": round(chance_tail(rates, branched), 3)}


def unrelated_pairs(results: list[dict]) -> list[tuple[dict, str, dict]]:
    """(neuron result, pair key, MLP pair) for every pair tested at formation and not adjacent at the end."""
    return [(r, key, p) for r in results for key, p in r["mlp"]["pairs"].items() if p.get("window") and not p["v2_adjacent"]]


def same_token_check(candidates: list[dict], results: list[dict], corpus: np.ndarray) -> dict:
    """Exploratory, added after reading the strongest branched pair (two facets firing on the same token).
    Splits the unrelated MLP pairs by whether their facets' most common activating token is the same, and,
    added after review, by whether they share any activating token; and compares early overlap (steps 128
    to 512) of all pairs that share a token with those that don't, which tests whether early drivers follow
    the token."""
    tokens = pair_tokens(candidates, corpus)

    def facets(r, key):
        return [tokens[r["neuron"], int(f)] for f in key.split("-")]

    def same_top(r, key):
        a, b = facets(r, key)
        return Counter(a).most_common(1)[0][0] == Counter(b).most_common(1)[0][0]

    def any_shared(r, key):
        a, b = facets(r, key)
        return bool(set(a) & set(b))

    pairs = unrelated_pairs(results)
    out = {"same top token": chance_split([p for r, k, p in pairs if same_top(r, k)]),
           "different top tokens": chance_split([p for r, k, p in pairs if not same_top(r, k)]),
           "share any token": chance_split([p for r, k, p in pairs if any_shared(r, k)]),
           "share no token": chance_split([p for r, k, p in pairs if not any_shared(r, k)])}
    early = {True: [], False: []}
    for r in results:
        for key, p in r["mlp"]["pairs"].items():
            early[any_shared(r, key)].append(float(np.mean(p["series"][:3])))
    out["median_overlap_steps_128_to_512"] = {"pairs sharing a token": [round(float(np.median(early[True])), 3), len(early[True])],
                                              "pairs sharing none": [round(float(np.median(early[False])), 3), len(early[False])]}
    return out


def v6_relation_check(candidates: list[dict], results: list[dict]) -> dict:
    """Exploratory, added after review: v13's "unrelated" means not adjacent by drivers at the end. This splits
    those pairs by whether v6's reader itself calls the two facets unrelated (centroid similarity below its
    unrelated level, v6's rule), the direct test of unrelated jobs, and gives how many branched pairs the
    v6-unrelated group would need for a chance tail below 0.05."""
    layer = Layer(MODEL, DEPTH)
    v6_unrelated = {}
    for c in candidates:
        s = c["sample"]
        units, _, levels = layer.embedded(PRIMARY, layer.rows[s], [s])
        facets = layer.neuron(s)["facets"]
        assert [len(f) for f in facets] == [len(f) for f in c["facets"]], "v6's facets should reproduce"
        centroids = [units[f].mean(axis=0) / np.linalg.norm(units[f].mean(axis=0)) for f in facets]
        for i, j in combinations(range(len(facets)), 2):
            v6_unrelated[c["neuron"], f"{i}-{j}"] = bool(centroids[i] @ centroids[j] < levels["unrelated"])
    pairs = unrelated_pairs(results)
    group = [p for r, k, p in pairs if v6_unrelated[r["neuron"], k]]
    rates = [p["null_exceedances"] / p["null_size"] for p in group]
    needed = next(k for k in range(len(rates) + 1) if chance_tail(rates, k) < 0.05)
    every = [(r, k, p) for r in results for k, p in r["mlp"]["pairs"].items()]
    return {"pairs_v6_calls_unrelated": sum(v6_unrelated[r["neuron"], k] for r, k, _ in every),
            "of_those_adjacent_at_end": sum(v6_unrelated[r["neuron"], k] and p["v2_adjacent"] for r, k, p in every),
            "v6 unrelated": {**chance_split(group), "branched_needed_for_tail_below_0.05": needed},
            "v6 related": chance_split([p for r, k, p in pairs if not v6_unrelated[r["neuron"], k]]),
            "branched_pairs_v6_related": sorted(f"{r['neuron']} {k}" for r, k, p in pairs
                                                if p["branched_then_separated"] and not v6_unrelated[r["neuron"], k])}


def robustness_check(results: list[dict]) -> dict:
    """Exploratory, added after review: how many branched pairs separate clearly at the end (final overlap at
    least 0.02 below the end threshold, or at or below the null's 95th percentile at each of the last three
    checkpoints), how many own pairs are adjacent at formation against chance, and how far above the end
    threshold the related pairs sit."""
    branched = [(r, p) for r in results for p in r["mlp"]["pairs"].values() if p.get("branched_then_separated")]
    tested = [p for r in results for p in r["mlp"]["pairs"].values() if p.get("window")]
    related = [(r, p) for r in results for p in r["mlp"]["pairs"].values() if p["v2_adjacent"]]
    return {"branched": len(branched),
            "branched_final_overlap_0.02_below_end_threshold": sum(p["series"][-1] <= r["end_threshold"] - 0.02 for r, p in branched),
            "branched_below_null_at_last_3_checkpoints": sum(all(x <= y for x, y in zip(p["series"][-3:], p["null_p95_series"][-3:]))
                                                             for _, p in branched),
            "own_pairs_adjacent_at_formation": f"{sum(p['adjacent'] for p in tested)} of {len(tested)}",
            "own_pairs_adjacent_at_formation_expected": round(sum(p["null_exceedances"] / p["null_size"] for p in tested), 1),
            "related_median_final_overlap": round(float(np.median([p["series"][-1] for _, p in related])), 3),
            "related_median_end_threshold": round(float(np.median([r["end_threshold"] for r, _ in related])), 3)}


def facet_sizes(candidates: list[dict]) -> dict:
    v4_candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    v4 = json.loads((RESULTS_DIR / "classification_v4.json").read_text())["neurons"]
    return {"v13_median_facet_size": float(np.median([len(f) for c in candidates for f in c["facets"]])),
            "v4_median_facet_size": float(np.median([len(f) for c in v4_candidates for f in c["facets"]])),
            "v4_related_median_final_overlap": round(float(np.median([p["series"][-1] for r in v4 for p in r["mlp"]["pairs"].values()
                                                                      if p["v2_adjacent"]])), 3)}


def analyze(candidates: list[dict], steps: list[dict]) -> tuple[list[dict], dict]:
    keys = [(c["neuron"], f) for c in candidates for f in range(len(c["facets"]))]
    results = []
    for c in candidates:
        n, n_facets = c["neuron"], len(c["facets"])
        windows = [formation_window(np.array([s["gains"][n, f] for s in steps])) for f in range(n_facets)]
        emergence = [None if w is None else STEPS[w[0]] for w in windows]
        overlaps, adjacent, end_threshold = final_adjacency(steps[-1], candidates, n)
        v1_like = {"hand_picked": False, "label": "n/a", "emergence_steps": emergence}
        v2_like = {"v2_label": "n/a", "adjacent_pairs": adjacent, "driver_overlaps": overlaps}
        results.append(neuron_results(n, n_facets, keys, steps, v1_like, v2_like))
        results[-1]["end_threshold"] = round(end_threshold, 4)
    summary = {"facets": dict(Counter(len(c["facets"]) for c in candidates)),
               "largest_reconstruction_error": max(s["reconstruction_error"] for s in steps)}
    for kind in ("mlp", "heads"):
        pairs = [p for r in results for p in r[kind]["pairs"].values()]
        tested = [p for p in pairs if p.get("window")]
        unrelated = [p for p in tested if not p["v2_adjacent"]]
        related = [p for p in tested if p["v2_adjacent"]]
        chance_keys = {label for r in results for label in r[kind]["chance_labels"]}
        summary[kind] = {
            "labels": dict(Counter(r[kind]["label"] for r in results)),
            "expected_labels_by_chance": {label: round(sum(r[kind]["chance_labels"].get(label, 0) for r in results), 1)
                                          for label in sorted(chance_keys)},
            "pairs_tested": len(tested), "unrelated_pairs_tested": len(unrelated),
            "branched_then_separated": sum(p["branched_then_separated"] for p in pairs),
            "branched_then_separated_expected_by_chance": round(sum(p["null_exceedances"] / p["null_size"] for p in unrelated), 1),
            "exploratory_chance_tail": round(chance_tail([p["null_exceedances"] / p["null_size"] for p in unrelated],
                                                         sum(p["branched_then_separated"] for p in pairs)), 3),
            "related_pairs_adjacent_at_formation": f"{sum(p['adjacent'] for p in related)} of {len(related)}",
            "branched_then_separated_neurons": sorted({r["neuron"] for r in results
                                                       if any(p["branched_then_separated"] for p in r[kind]["pairs"].values())}),
        }
    summary["share_of_pairs_adjacent_at_end"] = round(float(np.mean([p["v2_adjacent"] for r in results for p in r["mlp"]["pairs"].values()])), 3)
    v4 = json.loads((RESULTS_DIR / "classification_v4.json").read_text())["summary"]
    summary["v4_for_comparison"] = {"labels": v4["labels"]["all: v4 mlp"], "mlp": {k: v4["mlp"][k] for k in (
        "branched_then_separated", "branched_then_separated_expected_by_chance", "unrelated_pairs_tested",
        "related_pairs_adjacent_at_formation")}}
    return results, summary


def main() -> None:
    candidates = load_candidates()
    steps = [step_summary(step, candidates, path=v13_path(step)) for step in STEPS]
    results, summary = analyze(candidates, steps)
    summary["exploratory_same_token"] = same_token_check(candidates, results, load_corpus())
    summary["exploratory_after_review"] = {"v6_relation": v6_relation_check(candidates, results),
                                           "robustness": robustness_check(results), "facet_sizes": facet_sizes(candidates)}
    (RESULTS_DIR / "classification_v13.json").write_text(json.dumps({"summary": summary, "neurons": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
