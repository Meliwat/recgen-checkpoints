"""Apply the pre-registered v2 rules (recgen/rubric_v2.py) and compare with v1.

Reads cached data only: data/step143000_attribution.npz (recgen.attribution),
results/bridges.json and data/dense/ (recgen.bridges), and v1's results.
"""

import json
from collections import Counter

import numpy as np

from recgen.analyze import HAND_PICKED_PATH
from recgen.attribution import ATTRIBUTION_PATH
from recgen.bridges import BRIDGES_PATH, dense_path, facet_examples, window_steps
from recgen.config import CHECKPOINT_STEPS, RESULTS_DIR, SEED
from recgen.rubric import RECURSIVE_GENERALIZATION
from recgen.rubric_v2 import (
    BRIDGE_LEADS, BRIDGE_WITH_OR_AFTER, NO_BRIDGE_RESPONSE, NULL_PERCENTILE, SENSITIVITY_NULL_PERCENTILE,
    SENSITIVITY_TOP_UNITS, TOP_UNITS, WINDOW_TOO_SHORT, adjacency, bridge_lead, bridge_outcome, classify,
    drivers, null_threshold, overlap,
)

OUTCOME_ORDER = [BRIDGE_LEADS, BRIDGE_WITH_OR_AFTER, NO_BRIDGE_RESPONSE, WINDOW_TOO_SHORT]


def load_attribution():
    data = np.load(ATTRIBUTION_PATH)
    n_facet = int(data["n_facet_examples"])
    # Per-example activations divided by the LayerNorm scale: what each upstream unit's term is linear in.
    scaled = data["acts"].reshape(len(data["sigma"]), -1).astype(np.float32) / data["sigma"][:, None]
    parts = {k.removeprefix("part_"): data[k] for k in data.files if k.startswith("part_")}
    return {"examples": [tuple(e) for e in data["examples"][:n_facet]], "scaled": scaled, "n_facet": n_facet,
            "coefficients": data["coefficients"].reshape(scaled.shape[1], -1), "neurons": data["neurons"].tolist(),
            "parts": parts, "pre": data["pre"]}


def facet_shifts(att, candidates):
    """Each facet's mean scaled activation minus the background's, keyed (neuron, facet)."""
    row = {e: i for i, e in enumerate(att["examples"])}
    background = att["scaled"][att["n_facet"]:].mean(axis=0)
    shifts, rows = {}, {}
    for c in candidates:
        for f, facet in enumerate(c["facets"]):
            rows[c["neuron"], f] = [row[(w, p)] for w, p, _ in facet]
            shifts[c["neuron"], f] = att["scaled"][rows[c["neuron"], f]].mean(axis=0) - background
    return shifts, rows, background


def driver_overlaps(att, shifts, neuron: int, k: int):
    """Overlaps between this neuron's facets, and its null: its facets against every other
    neuron's facets, all read through this neuron's coefficients."""
    coefficient = att["coefficients"][:, att["neurons"].index(neuron)]
    tops = {key: drivers(shift * coefficient, k) for key, shift in shifts.items()}
    own = sorted(key for key in shifts if key[0] == neuron)
    pairs = np.array([[overlap(tops[a], tops[b], k) for b in own] for a in own])
    null = [overlap(tops[a], tops[x], k) for a in own for x in shifts if x[0] != neuron]
    return pairs, null


def split_half_overlap(att, rows, background, neuron: int, facet: int, k: int, rng) -> float:
    """Ceiling: overlap between the drivers of two random halves of one facet's examples."""
    coefficient = att["coefficients"][:, att["neurons"].index(neuron)]
    order = rng.permutation(rows[neuron, facet])
    halves = order[:len(order) // 2], order[len(order) // 2:]
    a, b = (drivers((att["scaled"][h].mean(axis=0) - background) * coefficient, k) for h in halves)
    return overlap(a, b, k)


def source_shares(att, rows, neuron: int, facet: int) -> dict[str, float]:
    """How the facet's rise in pre-activation over the background splits across sources."""
    k = att["neurons"].index(neuron)
    rise = {name: float(part[rows[neuron, facet], k].mean() - part[att["n_facet"]:, k].mean())
            for name, part in att["parts"].items()}
    total = sum(rise.values())
    return {name: value / total for name, value in rise.items()}


def adjacency_results(candidates, v1):
    att = load_attribution()
    shifts, rows, background = facet_shifts(att, candidates)
    rng = np.random.default_rng(SEED)
    neurons, variants = [], {}
    for r in v1["neurons"]:
        n = r["neuron"]
        times = [None if s is None else CHECKPOINT_STEPS.index(s) for s in r["emergence_steps"]]
        overlaps = {}
        for k in sorted({TOP_UNITS, *SENSITIVITY_TOP_UNITS}):
            pairs, null = driver_overlaps(att, shifts, n, k)
            overlaps[k] = pairs
            for percentile in (NULL_PERCENTILE, SENSITIVITY_NULL_PERCENTILE):
                label = classify(times, adjacency(pairs, null_threshold(null, percentile)))
                variants.setdefault(f"top {k}, null {percentile}th", Counter())[label] += 1
                if (k, percentile) == (TOP_UNITS, NULL_PERCENTILE):
                    primary = {"label": label, "threshold": null_threshold(null), "null": null}
        n_facets = len(r["emergence_steps"])
        neurons.append({
            "neuron": n,
            "hand_picked": r["hand_picked"],
            "v1_label": r["label"],
            "v2_label": primary["label"],
            "emergence_steps": r["emergence_steps"],
            "adjacency_threshold": round(primary["threshold"], 3),
            "null_median": round(float(np.median(primary["null"])), 3),
            "driver_overlaps": {f"{i}-{j}": round(float(overlaps[TOP_UNITS][i, j]), 3)
                                for i in range(n_facets) for j in range(i + 1, n_facets)},
            "split_half_overlaps": [round(split_half_overlap(att, rows, background, n, f, TOP_UNITS, rng), 3)
                                    for f in range(n_facets)],
            "mlp_neuron_share": [round(source_shares(att, rows, n, f)["mlp_neurons"], 3) for f in range(n_facets)],
        })
    reconstruction_error = float(np.abs(sum(att["parts"].values()) - att["pre"]).max())
    return neurons, variants, reconstruction_error


def pair_gains(pair: dict, facet_row: dict) -> dict[str, np.ndarray]:
    """Pre-activation gains over the pair's window (see rubric_v2): the later facet's examples
    over the pool, the bridge over its A-matched control, and the placebo over the pool."""
    gains = {"facet": [], "bridge": [], "placebo": []}
    for step in window_steps(pair["step_a"], pair["step_b"]):
        data = np.load(dense_path(step))
        k = data["neurons"].tolist().index(pair["neuron"])
        pool = data["pool"][:, k].astype(np.float32)
        facet = data["facet"][facet_row[pair["neuron"], pair["b"]], k].astype(np.float32)
        baseline = np.median(pool)
        gains["facet"].append(np.median(facet) - baseline)
        gains["bridge"].append(np.median(pool[pair["bridge"]]) - np.median(pool[pair["a_control"]]))
        gains["placebo"].append(np.median(pool[pair["b_control"]]) - baseline)
    return {name: np.array(g) for name, g in gains.items()}


def bridge_results(candidates):
    pairs = json.loads(BRIDGES_PATH.read_text())
    row = {e: i for i, e in enumerate(facet_examples(candidates))}
    facet_row = {(c["neuron"], f): [row[(w, p)] for w, p, _ in facet]
                 for c in candidates for f, facet in enumerate(c["facets"])}
    results = []
    for pair in pairs:
        gains = pair_gains(pair, facet_row)
        results.append({
            **{k: pair[k] for k in ("neuron", "a", "b", "step_a", "step_b", "similarity", "bridge_examples")},
            "steps": window_steps(pair["step_a"], pair["step_b"]),
            "gains": {name: [round(float(x), 3) for x in g] for name, g in gains.items()},
            "bridge": bridge_outcome(gains["facet"], gains["bridge"]),
            "placebo": bridge_outcome(gains["facet"], gains["placebo"]),
            "bridge_lead": bridge_lead(gains["facet"], gains["bridge"]),
            "placebo_lead": bridge_lead(gains["facet"], gains["placebo"]),
        })
    return results


def later_facets_with_bridge(bridges) -> dict[str, int]:
    """Later facets with a leading bridge from at least one earlier facet, among those with a testable pair."""
    testable, leading = set(), set()
    for b in bridges:
        key = (b["neuron"], b["b"])
        if b["bridge"] != WINDOW_TOO_SHORT:
            testable.add(key)
        if b["bridge"] == BRIDGE_LEADS:
            leading.add(key)
    return {"testable": len(testable), "with a leading bridge": len(leading)}


def write_report(neurons, bridges, hand_picked) -> None:
    lines = ["# v2 results per neuron\n",
             "Driver overlaps are the share of the top 100 upstream drivers two facets have in common; "
             "facets are adjacent above the neuron's threshold (95th percentile of its null). "
             "Bridge outcomes are per (earlier facet, later facet) pair.\n"]
    by_pair = {}
    for b in bridges:
        by_pair.setdefault(b["neuron"], []).append(b)
    for r in neurons:
        n = r["neuron"]
        names = hand_picked.get(str(n), {}).get("facets") or [f"facet {i}" for i in range(len(r["emergence_steps"]))]
        lines.append(f"## Neuron {n}: v1 {r['v1_label']}, v2 {r['v2_label']}\n")
        lines += [f"- facet {i}, {name}: emerges at step {s}, split-half overlap {h}"
                  for i, (name, s, h) in enumerate(zip(names, r["emergence_steps"], r["split_half_overlaps"]))]
        lines.append(f"- threshold {r['adjacency_threshold']} (null median {r['null_median']}); overlaps "
                     + ", ".join(f"{k}: {v}" for k, v in r["driver_overlaps"].items()) + "\n")
        for b in by_pair.get(n, []):
            sim = b["similarity"]["bridge"]
            lines.append(f"**Bridge {b['a']} → {b['b']}** (steps {b['step_a']} to {b['step_b']}, "
                         f"{len(b['steps'])} checkpoints): {b['bridge']}; placebo: {b['placebo']}. "
                         f"Bridge similarity to A {sim['to_a']}, to B {sim['to_b']}.\n")
            lines += [f"- `{text}`" for text in b["bridge_examples"]]
            lines.append("")
    (RESULTS_DIR / "v2_neurons.md").write_text("\n".join(lines))


def main() -> None:
    candidates = json.loads((RESULTS_DIR / "candidates.json").read_text())
    v1 = json.loads((RESULTS_DIR / "classification.json").read_text())
    hand_picked = json.loads(HAND_PICKED_PATH.read_text())
    neurons, variants, reconstruction_error = adjacency_results(candidates, v1)
    bridges = bridge_results(candidates)
    hand = [r for r in neurons if r["hand_picked"]]
    summary = {
        "reconstruction_error": reconstruction_error,
        "labels_v1_all": Counter(r["v1_label"] for r in neurons),
        "labels_v2_all": Counter(r["v2_label"] for r in neurons),
        "labels_v1_hand_picked": Counter(r["v1_label"] for r in hand),
        "labels_v2_hand_picked": Counter(r["v2_label"] for r in hand),
        "label_changes": Counter(f"{r['v1_label']} -> {r['v2_label']}" for r in neurons),
        "sensitivity": variants,
        "median_split_half_overlap": float(np.median([x for r in neurons for x in r["split_half_overlaps"]])),
        "median_facet_pair_overlap": float(np.median([x for r in neurons for x in r["driver_overlaps"].values()])),
        "median_null_overlap": float(np.median([r["null_median"] for r in neurons])),
        "median_adjacency_threshold": float(np.median([r["adjacency_threshold"] for r in neurons])),
        "median_mlp_neuron_share": float(np.median([x for r in neurons for x in r["mlp_neuron_share"]])),
        "bridge_outcomes": Counter(b["bridge"] for b in bridges),
        "placebo_outcomes": Counter(b["placebo"] for b in bridges),
        "later_facets": later_facets_with_bridge(bridges),
        "bridge_leads": sorted(b["bridge_lead"] for b in bridges if b["bridge"] != WINDOW_TOO_SHORT
                               and b["bridge_lead"] is not None),
        "placebo_leads": sorted(b["placebo_lead"] for b in bridges if b["placebo"] != WINDOW_TOO_SHORT
                                and b["placebo_lead"] is not None),
        "recursive_generalization_v2_with_leading_bridge": sorted(
            {b["neuron"] for b in bridges if b["bridge"] == BRIDGE_LEADS}
            & {r["neuron"] for r in neurons if r["v2_label"] == RECURSIVE_GENERALIZATION}),
    }
    (RESULTS_DIR / "classification_v2.json").write_text(
        json.dumps({"summary": summary, "neurons": neurons, "bridges": bridges}, indent=1))
    write_report(neurons, bridges, hand_picked)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
