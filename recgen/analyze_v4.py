"""Apply the pre-registered v4 rules (recgen/rubric_v4.py) and compare with v1 and v2.

Reads cached data only: data/v4/step*.npz (recgen.formation) and v1's and v2's results.
"""

import json
from collections import Counter
from itertools import combinations

import numpy as np

from recgen.analyze import HAND_PICKED_PATH
from recgen.analyze_v2 import chance_labels
from recgen.config import CHECKPOINT_STEPS, RESULTS_DIR
from recgen.formation import formation_path
from recgen.rubric_v2 import NULL_PERCENTILE, classify, drivers, overlap
from recgen.rubric_v4 import (
    STEPS, TOP_HEADS, TOP_UNITS, branched_then_separated, formation_threshold, formation_window, pair_window, peak,
)

PARTS = ["mlp_neurons", "attention_heads", "embedding", "mlp_biases", "attention_biases", "constant"]
KINDS = {"mlp": TOP_UNITS, "heads": TOP_HEADS}


def step_summary(step: int, candidates: list[dict]) -> dict:
    """One checkpoint: driver sets of every facet of every traced neuron, read through each traced
    neuron's weights (MLP neurons and heads), plus each facet's gain and source shares on its own neuron."""
    data = np.load(formation_path(step))
    n_facet = int(data["n_facet_examples"])
    row = {tuple(e): i for i, e in enumerate(data["examples"][:n_facet])}
    neurons = data["neurons"].tolist()
    scaled = data["acts"].reshape(len(data["sigma"]), -1).astype(np.float32) / data["sigma"][:, None]
    heads = data["heads"].reshape(len(data["sigma"]), -1, len(neurons))
    coefficients = data["coefficients"].reshape(scaled.shape[1], -1)
    rows = {(c["neuron"], f): [row[(w, p)] for w, p, _ in facet] for c in candidates for f, facet in enumerate(c["facets"])}
    shifts = {key: scaled[r].mean(axis=0) - scaled[n_facet:].mean(axis=0) for key, r in rows.items()}
    head_shifts = {key: heads[r].mean(axis=0) - heads[n_facet:].mean(axis=0) for key, r in rows.items()}
    summary = {"mlp": {}, "heads": {}, "gains": {}, "shares": {}, "reconstruction_error": float(data["reconstruction_error"])}
    for k, n in enumerate(neurons):
        for key in rows:
            summary["mlp"][n, key] = drivers(shifts[key] * coefficients[:, k], TOP_UNITS)
            summary["heads"][n, key] = drivers(head_shifts[key][:, k], TOP_HEADS)
    for (n, f), r in rows.items():
        k = neurons.index(n)
        summary["gains"][n, f] = float(np.median(data["pre"][r, k]) - np.median(data["pre"][n_facet:, k]))
        rise = {p: float(data[f"part_{p}"][r, k].mean() - data[f"part_{p}"][n_facet:, k].mean()) for p in PARTS}
        summary["shares"][n, f] = {p: v / sum(rise.values()) for p, v in rise.items()}
    return summary


def overlap_series(steps: list[dict], kind: str, neuron: int, a, b) -> np.ndarray:
    return np.array([overlap(s[kind][neuron, a], s[kind][neuron, b], KINDS[kind]) for s in steps])


def evaluate_pair(series: np.ndarray, null: list[np.ndarray], window: tuple[int, int]) -> dict:
    """The rubric's test for one pair, plus what the report and figures need."""
    threshold = formation_threshold(null, window)
    return {
        "window": [STEPS[window[0]], STEPS[window[1]]],
        "peak": round(peak(series, window), 3),
        "threshold": round(threshold, 3),
        "adjacent": peak(series, window) > threshold,
        "null_exceedances": int(sum(peak(s, window) > threshold for s in null)),
        "null_size": len(null),
        "series": [round(float(x), 3) for x in series],
        "null_p95_series": [round(float(x), 3) for x in np.percentile(np.array(null), NULL_PERCENTILE, axis=0)],
    }


def neuron_results(n: int, n_facets: int, facet_keys: list, steps: list[dict], v1_row: dict, v2_row: dict) -> dict:
    gains = {f: np.array([s["gains"][n, f] for s in steps]) for f in range(n_facets)}
    windows = {f: formation_window(gains[f]) for f in range(n_facets)}
    others = [key for key in facet_keys if key[0] != n]
    times = [None if s is None else CHECKPOINT_STEPS.index(s) for s in v1_row["emergence_steps"]]
    result = {
        "neuron": n, "hand_picked": v1_row["hand_picked"], "v1_label": v1_row["label"], "v2_label": v2_row["v2_label"],
        "emergence_steps": v1_row["emergence_steps"],
        "formation_windows": [None if w is None else [STEPS[w[0]], STEPS[w[1]]] for w in windows.values()],
        "gains": [[round(float(x), 3) for x in g] for g in gains.values()],
    }
    for kind in KINDS:
        adjacent = np.eye(n_facets, dtype=bool)
        pairs, exceed, tested = {}, 0, 0
        for i, j in combinations(range(n_facets), 2):
            window = pair_window(windows[i], windows[j])
            series = overlap_series(steps, kind, n, (n, i), (n, j))
            if window is None:
                pairs[f"{i}-{j}"] = {"window": None, "adjacent": False, "series": [round(float(x), 3) for x in series]}
                continue
            null = [overlap_series(steps, kind, n, (n, a), x) for a in (i, j) for x in others]
            pairs[f"{i}-{j}"] = evaluate_pair(series, null, window)
            adjacent[i, j] = adjacent[j, i] = pairs[f"{i}-{j}"]["adjacent"]
            exceed += pairs[f"{i}-{j}"]["null_exceedances"]
            tested += pairs[f"{i}-{j}"]["null_size"]
        rate = exceed / tested if tested else 0.0
        for key, pair in pairs.items():
            pair["v2_adjacent"] = v2_row["adjacent_pairs"][key]["v2"]
            pair["branched_then_separated"] = branched_then_separated(pair["adjacent"], pair["v2_adjacent"])
        result[kind] = {"label": classify(times, adjacent), "pairs": pairs, "null_exceedance": round(rate, 4),
                        "chance_labels": chance_labels(times, rate)}
    result["final_mlp_overlap_vs_v2"] = max(abs(p["series"][-1] - v2_row["driver_overlaps"][key])
                                            for key, p in result["mlp"]["pairs"].items())
    return result


def least_related_pair(v1_row: dict) -> str:
    return min(v1_row["facet_similarities"], key=v1_row["facet_similarities"].get)


def label_counts(results: list[dict]) -> dict[str, Counter]:
    labels = {}
    for group_name, group in (("all", results), ("hand_picked", [r for r in results if r["hand_picked"]])):
        labels[f"{group_name}: v1"] = Counter(r["v1_label"] for r in group)
        labels[f"{group_name}: v2"] = Counter(r["v2_label"] for r in group)
        for kind in KINDS:
            labels[f"{group_name}: v4 {kind}"] = Counter(r[kind]["label"] for r in group)
    return labels


def summarize(results: list[dict], steps: list[dict], v1_rows: dict) -> dict:
    summary = {
        "largest_reconstruction_error": max(s["reconstruction_error"] for s in steps),
        "largest_final_mlp_overlap_difference_from_v2": max(r["final_mlp_overlap_vs_v2"] for r in results),
        "facets_without_window": sum(w is None for r in results for w in r["formation_windows"]),
        "window_lengths_in_checkpoints": Counter(STEPS.index(w[1]) - STEPS.index(w[0]) + 1
                                                 for r in results for w in r["formation_windows"] if w),
        "labels": label_counts(results),
    }
    for kind in KINDS:
        pairs = [p for r in results for p in r[kind]["pairs"].values()]
        unrelated = [p for p in pairs if not p["v2_adjacent"] and p.get("window")]
        related = [p for p in pairs if p["v2_adjacent"] and p.get("window")]
        chance_keys = {label for r in results for label in r[kind]["chance_labels"]}
        summary[kind] = {
            "expected_labels_by_chance": {label: round(sum(r[kind]["chance_labels"].get(label, 0) for r in results), 1)
                                          for label in sorted(chance_keys)},
            "median_null_exceedance": float(np.median([r[kind]["null_exceedance"] for r in results])),
            "pairs_tested": sum(1 for p in pairs if p.get("window")),
            "branched_then_separated": sum(p["branched_then_separated"] for p in pairs),
            "branched_then_separated_expected_by_chance": round(sum(p["null_exceedances"] / p["null_size"] for p in unrelated), 1),
            "unrelated_pairs_tested": len(unrelated),
            "related_pairs_adjacent_at_formation": f"{sum(p['adjacent'] for p in related)} of {len(related)}",
            "branched_then_separated_neurons": sorted({r["neuron"] for r in results
                                                       if any(p["branched_then_separated"] for p in r[kind]["pairs"].values())}),
        }
    summary["least_related_pair_mlp_peak_vs_final"] = {
        "median_peak_in_window": float(np.nanmedian([r["mlp"]["pairs"][least_related_pair(v1_rows[r["neuron"]])].get("peak", np.nan)
                                                     for r in results])),
        "median_final": float(np.median([r["mlp"]["pairs"][least_related_pair(v1_rows[r["neuron"]])]["series"][-1]
                                         for r in results])),
    }
    summary["median_source_share_by_step"] = {
        part: {str(step): round(float(np.median([v[part] for v in s["shares"].values()])), 3) for step, s in zip(STEPS, steps)}
        for part in ("mlp_neurons", "attention_heads", "embedding")
    }
    return summary


def write_report(results: list[dict], hand_picked: dict, v1_rows: dict) -> None:
    lines = ["# v4 results per neuron\n",
             "For each facet pair: the formation window (steps), the largest driver overlap inside it (peak), "
             "the threshold (95th percentile of the null's peaks in the same window), and whether the pair is "
             "adjacent at formation and at the final checkpoint (v2). MLP-neuron drivers are the primary measure.\n"]
    for r in results:
        n = r["neuron"]
        names = hand_picked.get(str(n), {}).get("facets") or [f"facet {i}" for i in range(len(r["emergence_steps"]))]
        lines.append(f"## Neuron {n}: v1 {r['v1_label']}, v2 {r['v2_label']}, v4 {r['mlp']['label']} "
                     f"(heads: {r['heads']['label']})\n")
        lines += [f"- facet {i}, {name}: v1 emergence {s}, forms over {w}" for i, (name, s, w)
                  in enumerate(zip(names, r["emergence_steps"], r["formation_windows"]))]
        lines.append(f"- least-related pair: {least_related_pair(v1_rows[n])}\n")
        for kind in KINDS:
            for key, p in r[kind]["pairs"].items():
                if not p.get("window"):
                    lines.append(f"- {kind} {key}: no formation window")
                    continue
                lines.append(f"- {kind} {key}: window {p['window'][0]} to {p['window'][1]}, peak {p['peak']} vs threshold "
                             f"{p['threshold']}: {'adjacent' if p['adjacent'] else 'not adjacent'} at formation, "
                             f"{'adjacent' if p['v2_adjacent'] else 'not adjacent'} at the end"
                             + (" (branched then separated)" if p["branched_then_separated"] else ""))
        lines.append("")
    (RESULTS_DIR / "v4_neurons.md").write_text("\n".join(lines))


def main() -> None:
    candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    v1_rows = {r["neuron"]: r for r in json.loads((RESULTS_DIR / "classification.json").read_text())["neurons"]}
    v2_rows = {r["neuron"]: r for r in json.loads((RESULTS_DIR / "classification_v2.json").read_text())["neurons"]}
    hand_picked = json.loads(HAND_PICKED_PATH.read_text())
    steps = [step_summary(step, candidates) for step in STEPS]
    facet_keys = [(c["neuron"], f) for c in candidates for f in range(len(c["facets"]))]
    results = [neuron_results(c["neuron"], len(c["facets"]), facet_keys, steps, v1_rows[c["neuron"]], v2_rows[c["neuron"]])
               for c in candidates]
    summary = summarize(results, steps, v1_rows)
    (RESULTS_DIR / "classification_v4.json").write_text(json.dumps({"summary": summary, "neurons": results}, indent=1))
    write_report(results, hand_picked, v1_rows)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
