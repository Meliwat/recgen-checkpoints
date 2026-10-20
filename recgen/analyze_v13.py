"""Apply the pre-registered v13 rules (recgen/rubric_v13.py): v4's formation-time test on network-native
facets of neurons the network reader calls mixed. Reads data/v13/ (recgen.record_v13) and v4's results.
"""

import json
from collections import Counter
from itertools import combinations

import numpy as np

from recgen.analyze_v4 import neuron_results, step_summary
from recgen.candidates_v13 import load_candidates, v13_path
from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.rubric_v2 import null_threshold, overlap
from recgen.rubric_v4 import STEPS, TOP_UNITS, formation_window


def final_adjacency(final: dict, candidates: list[dict], n: int) -> tuple[dict, dict]:
    """v2's rule at the final checkpoint: each pair's MLP driver overlap, and whether it beats the 95th
    percentile of this neuron's null (its facets against the other neurons' facets)."""
    own = [key for key in final["mlp"] if key[0] == n and key[1][0] == n]
    keys = [(c["neuron"], f) for c in candidates for f in range(len(c["facets"]))]
    null = [overlap(final["mlp"][n, a], final["mlp"][n, x], TOP_UNITS) for a in (k[1] for k in own) for x in keys if x[0] != n]
    threshold = null_threshold(null)
    overlaps, adjacent = {}, {}
    for (_, a), (_, b) in combinations(sorted(own), 2):
        key = f"{a[1]}-{b[1]}"
        overlaps[key] = overlap(final["mlp"][n, a], final["mlp"][n, b], TOP_UNITS)
        adjacent[key] = {"v2": bool(overlaps[key] > threshold)}
    return overlaps, adjacent


def chance_tail(rates: list[float], observed: int) -> float:
    """Exploratory, added after seeing the counts: the probability of at least `observed` branched-then-separated
    pairs if each unrelated pair crossed its threshold only at its own null's false-positive rate, independently.
    Pairs within a neuron share facets, so this is approximate."""
    distribution = np.array([1.0])
    for rate in rates:
        distribution = np.convolve(distribution, [1 - rate, rate])
    return float(distribution[observed:].sum())


def same_token_check(candidates: list[dict], results: list[dict], corpus: np.ndarray) -> dict:
    """Exploratory, added after reading the strongest branched pair (two facets firing on the same token):
    early in training a neuron's drivers follow the input token, so facets that fire on the same token
    share drivers early whatever happens later. Splits the unrelated MLP pairs by whether their facets'
    most common activating token is the same."""
    top_token = {(c["neuron"], f): Counter(int(corpus[w, p]) for w, p, _ in facet).most_common(1)[0][0]
                 for c in candidates for f, facet in enumerate(c["facets"])}
    out = {}
    for same in (True, False):
        pairs = [p for r in results for key, p in r["mlp"]["pairs"].items()
                 if p.get("window") and not p["v2_adjacent"]
                 and (top_token[r["neuron"], int(key.split("-")[0])] == top_token[r["neuron"], int(key.split("-")[1])]) == same]
        rates = [p["null_exceedances"] / p["null_size"] for p in pairs]
        branched = sum(p["branched_then_separated"] for p in pairs)
        out["same top token" if same else "different top tokens"] = {
            "unrelated_pairs": len(pairs), "branched_then_separated": branched,
            "expected_by_chance": round(sum(rates), 1), "chance_tail": round(chance_tail(rates, branched), 3)}
    return out


def analyze(candidates: list[dict], steps: list[dict]) -> tuple[list[dict], dict]:
    keys = [(c["neuron"], f) for c in candidates for f in range(len(c["facets"]))]
    results = []
    for c in candidates:
        n, n_facets = c["neuron"], len(c["facets"])
        windows = [formation_window(np.array([s["gains"][n, f] for s in steps])) for f in range(n_facets)]
        emergence = [None if w is None else STEPS[w[0]] for w in windows]
        overlaps, adjacent = final_adjacency(steps[-1], candidates, n)
        v1_like = {"hand_picked": False, "label": "n/a", "emergence_steps": emergence}
        v2_like = {"v2_label": "n/a", "adjacent_pairs": adjacent, "driver_overlaps": overlaps}
        results.append(neuron_results(n, n_facets, keys, steps, v1_like, v2_like))
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
    (RESULTS_DIR / "classification_v13.json").write_text(json.dumps({"summary": summary, "neurons": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
