"""Apply the pre-registered v23 rules (recgen/rubric_v23.py): are weakly firing neurons' facets driven by
different upstream circuits more often than strongly firing neurons'? Reads data/v23/ (recgen.record_v23),
data/v13/'s final checkpoint, and v6's and v13's results. Writes results/classification_v23.json.
"""

import json
from collections import Counter

import numpy as np

from recgen.analyze_v4 import KINDS, step_summary
from recgen.analyze_v6 import Layer
from recgen.analyze_v13 import final_adjacency
from recgen.candidates_v13 import load_candidates as load_v13_candidates
from recgen.candidates_v13 import v13_path
from recgen.candidates_v23 import RECORDING_PATH, layer_quartiles, load_candidates
from recgen.config import RESULTS_DIR
from recgen.rubric_v23 import (
    DEPTH, MIN_FACETS, MODEL, STEP, control_reading, facet_count_strata, facet_size_strata, neuron_measures,
    paired_difference, reading, rng_for, spearman, top_minus_rest,
)

PURPOSE = {"mlp": 0, "heads": 100}  # bootstrap seeds: [SEED, 23, PURPOSE[kind] + test number]


def neuron_rows(candidates: list[dict], final: dict) -> list[dict]:
    """Per neuron: peak, facets and, for each driver kind, its pair overlaps, null threshold and measures."""
    rows = []
    for c in candidates:
        sizes = [len(f) for f in c["facets"]]
        row = {k: c[k] for k in ("neuron", "quartile", "peak", "log_peak", "unrelated_mixed", "reader_least_pair")}
        row.update({"n_facets": len(sizes), "facet_sizes": sizes, "median_facet_size": float(np.median(sizes))})
        for kind in KINDS:
            overlaps, _, threshold = final_adjacency(final, candidates, c["neuron"], kind)
            row[kind] = {"overlaps": overlaps, **neuron_measures(overlaps, threshold, c["reader_least_pair"])}
        rows.append(row)
    return rows


def column(rows: list[dict], kind: str, measure: str) -> np.ndarray:
    return np.array([float(r[kind][measure]) for r in rows])


def driver_tests(rows: list[dict], kind: str) -> dict:
    """P1 and P2, and S1 to S3, for one driver kind."""
    q = np.array([r["quartile"] for r in rows])
    peak = np.array([r["peak"] for r in rows])
    base = PURPOSE[kind]
    return {"P1_least_pair_unrelated": top_minus_rest(q, column(rows, kind, "least_unrelated"), rng_for(base + 1)),
            "P2_spearman_peak_margin": spearman(q, peak, column(rows, kind, "margin"), rng_for(base + 2)),
            "S1_share_unrelated": top_minus_rest(q, column(rows, kind, "share_unrelated"), rng_for(base + 4)),
            "S2_margin": top_minus_rest(q, column(rows, kind, "margin"), rng_for(base + 5)),
            "S3_reader_least_pair_unrelated": top_minus_rest(q, column(rows, kind, "reader_least_unrelated"), rng_for(base + 6))}


def pre_registered(rows: list[dict]) -> dict:
    q = np.array([r["quartile"] for r in rows])
    reader = np.array([float(r["unrelated_mixed"]) for r in rows])
    mlp = driver_tests(rows, "mlp")
    mlp["P3_minus_reader_flag"] = paired_difference(q, column(rows, "mlp", "least_unrelated"), reader, rng_for(3))
    mlp["reader_flag_top_minus_rest"] = top_minus_rest(q, reader, rng_for(7))
    mlp["reading"] = reading(mlp["P1_least_pair_unrelated"]["interval"], mlp["P2_spearman_peak_margin"]["interval"],
                             mlp["P3_minus_reader_flag"]["interval"])
    size_strata = facet_size_strata(np.array([r["median_facet_size"] for r in rows]))
    count_strata = facet_count_strata(np.array([r["n_facets"] for r in rows]))
    c1 = top_minus_rest(q, column(rows, "mlp", "margin"), rng_for(8), size_strata)
    c2 = top_minus_rest(q, column(rows, "mlp", "least_unrelated"), rng_for(9), count_strata)
    controls = {
        "C1_margin_within_facet_size_strata": {**c1, "reading": control_reading(mlp["S2_margin"]["interval"], c1["interval"]),
                                               "strata_by_quartile": strata_table(q, size_strata),
                                               "stratum_median_facet_sizes": stratum_ranges(rows, size_strata)},
        "C2_least_pair_within_facet_count_strata": {**c2, "reading": control_reading(mlp["P1_least_pair_unrelated"]["interval"],
                                                                                     c2["interval"]),
                                                    "strata_by_quartile": strata_table(q, count_strata)},
    }
    return {"mlp": mlp, "heads": driver_tests(rows, "heads"), "controls": controls}


def strata_table(q: np.ndarray, strata: np.ndarray) -> dict:
    return {str(int(s)): [int(np.sum((strata == s) & (q == k))) for k in range(4)] for s in np.unique(strata)}


def stratum_ranges(rows: list[dict], strata: np.ndarray) -> dict:
    sizes = np.array([r["median_facet_size"] for r in rows])
    return {str(int(s)): [float(sizes[strata == s].min()), float(sizes[strata == s].max())] for s in np.unique(strata)}


def by_quartile(rows: list[dict]) -> dict:
    """Descriptive, per peak quartile (weakest to strongest)."""
    out = {}
    for k in range(4):
        group = [r for r in rows if r["quartile"] == k]
        entry = {"neurons": len(group), "median_peak": round(float(np.median([r["peak"] for r in group])), 3),
                 "reader_flag_rate": round(float(np.mean([r["unrelated_mixed"] for r in group])), 3),
                 "median_facet_size": float(np.median([s for r in group for s in r["facet_sizes"]])),
                 "mean_facets": round(float(np.mean([r["n_facets"] for r in group])), 2)}
        for kind in KINDS:
            entry[kind] = {"least_pair_unrelated": round(float(np.mean([r[kind]["least_unrelated"] for r in group])), 3),
                           "share_unrelated": round(float(np.mean([r[kind]["share_unrelated"] for r in group])), 3),
                           "reader_least_pair_unrelated": round(float(np.mean([r[kind]["reader_least_unrelated"] for r in group])), 3),
                           "median_margin": round(float(np.median([r[kind]["margin"] for r in group])), 3),
                           "median_least_overlap": round(float(np.median([r[kind]["least_overlap"] for r in group])), 3),
                           "median_threshold": round(float(np.median([r[kind]["threshold"] for r in group])), 3)}
        out[str(k)] = entry
    return out


def eligible_counts() -> dict:
    _, _, quartile, n_facets = layer_quartiles()
    return {"sampled": [int(np.sum(quartile == k)) for k in range(4)],
            "eligible": [int(np.sum((quartile == k) & (n_facets >= MIN_FACETS))) for k in range(4)]}


def v13_check(rows: list[dict]) -> dict:
    """v23's code on v13's final checkpoint: it must reproduce v13's published end thresholds and final MLP
    overlaps; shared neurons must get the same overlaps from both recordings; and v13's 40 neurons get the
    primary measure by peak quartile (description only)."""
    candidates = load_v13_candidates()
    final = step_summary(STEP, candidates, path=v13_path(STEP))
    published = {r["neuron"]: r for r in json.loads((RESULTS_DIR / "classification_v13.json").read_text())["neurons"]}
    ids, peak, quartile, _ = layer_quartiles()
    peak_of, quartile_of = dict(zip(ids, peak)), dict(zip(ids, quartile))
    layer = Layer(MODEL, DEPTH)
    v23 = {r["neuron"]: r for r in rows}
    threshold_diff, overlap_diff, shared, v13_rows = 0.0, 0.0, {}, []
    for c in candidates:
        n = c["neuron"]
        overlaps, _, threshold = final_adjacency(final, candidates, n)
        threshold_diff = max(threshold_diff, abs(round(threshold, 4) - published[n]["end_threshold"]))
        overlap_diff = max(overlap_diff, max(abs(round(v, 3) - published[n]["mlp"]["pairs"][k]["series"][-1]) for k, v in overlaps.items()))
        i, j = layer.neuron(c["sample"])["least_related_pair"]
        v13_rows.append({"neuron": n, "quartile": int(quartile_of[n]), "peak": float(peak_of[n]),
                         **neuron_measures(overlaps, threshold, f"{i}-{j}")})
        if n in v23:
            head_overlaps = final_adjacency(final, candidates, n, "heads")[0]
            shared[str(n)] = {"mlp_largest_overlap_difference": max(abs(v - v23[n]["mlp"]["overlaps"][k]) for k, v in overlaps.items()),
                              "heads_largest_overlap_difference": max(abs(v - v23[n]["heads"]["overlaps"][k]) for k, v in head_overlaps.items()),
                              "v13_threshold": round(threshold, 4), "v23_threshold": round(v23[n]["mlp"]["threshold"], 4)}
    return {"largest_end_threshold_difference": threshold_diff, "largest_final_overlap_difference": overlap_diff,
            "shared_neurons": shared,
            "v13_neurons_by_quartile": {str(k): {"neurons": sum(r["quartile"] == k for r in v13_rows),
                                                 "least_pair_unrelated": sum(r["least_unrelated"] for r in v13_rows if r["quartile"] == k),
                                                 "median_margin": round(float(np.median([r["margin"] for r in v13_rows if r["quartile"] == k])), 3)
                                                 if any(r["quartile"] == k for r in v13_rows) else None}
                                        for k in range(4)}}


def rounded(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r = {**r, "peak": round(r["peak"], 4), "log_peak": round(r["log_peak"], 4)}
        for kind in KINDS:
            r[kind] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r[kind].items()}
            r[kind]["overlaps"] = {k: round(v, 3) for k, v in r[kind]["overlaps"].items()}
        out.append(r)
    return out


def main() -> None:
    candidates = load_candidates()
    final = step_summary(STEP, candidates, path=RECORDING_PATH)
    rows = neuron_rows(candidates, final)
    summary = {"neurons_by_quartile": eligible_counts() | {"selected": [sum(r["quartile"] == k for r in rows) for k in range(4)]},
               "facets": len([s for r in rows for s in r["facet_sizes"]]), "pairs": sum(len(r["mlp"]["overlaps"]) for r in rows),
               "facet_count_distribution": {str(k): v for k, v in sorted(Counter(r["n_facets"] for r in rows).items())},
               "reconstruction_error": final["reconstruction_error"],
               "pre_registered": pre_registered(rows), "by_quartile": by_quartile(rows),
               "share_of_pairs_unrelated": {kind: round(float(np.mean([o <= r[kind]["threshold"] for r in rows
                                                                       for o in r[kind]["overlaps"].values()])), 3) for kind in KINDS},
               "v13_check": v13_check(rows)}
    (RESULTS_DIR / "classification_v23.json").write_text(json.dumps({"summary": summary, "neurons": rounded(rows)}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
