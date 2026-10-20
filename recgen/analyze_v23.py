"""Apply the pre-registered v23 rules (recgen/rubric_v23.py): are weakly firing neurons' facets driven by
different upstream circuits more often than strongly firing neurons'? Reads data/v23/ (recgen.record_v23),
data/v13/'s final checkpoint, and v6's and v13's results. Writes results/classification_v23.json.
"""

import json
import pickle
from collections import Counter

import numpy as np

from recgen.analyze_v4 import KINDS, step_summary
from recgen.analyze_v6 import Layer
from recgen.analyze_v13 import final_adjacency
from recgen.candidates_v13 import load_candidates as load_v13_candidates
from recgen.candidates_v13 import v13_path
from recgen.candidates_v23 import RECORDING_PATH, V23_DIR, layer_quartiles, load_candidates
from scipy.stats import spearmanr

from recgen.config import RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.rubric_v2 import TOP_UNITS, drivers, overlap
from recgen.rubric_v23 import (
    DEPTH, MIN_FACETS, MODEL, STEP, control_reading, facet_count_strata, facet_size_strata, neuron_measures,
    paired_difference, reading, rng_for, spearman, top_minus_rest,
)

PURPOSE = {"mlp": 0, "heads": 100}  # bootstrap seeds: [SEED, 23, PURPOSE[kind] + test number]
SUMMARY_CACHE = V23_DIR / "step143000_summary.pkl"


def final_summary(candidates: list[dict]) -> dict:
    """analyze_v4.step_summary on v23's recording (about 15 minutes), cached in data/v23/."""
    if SUMMARY_CACHE.exists() and SUMMARY_CACHE.stat().st_mtime > RECORDING_PATH.stat().st_mtime:
        return pickle.loads(SUMMARY_CACHE.read_bytes())
    final = step_summary(STEP, candidates, path=RECORDING_PATH)
    SUMMARY_CACHE.write_bytes(pickle.dumps(final))
    return final


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


def reader_split(rows: list[dict]) -> dict:
    """Exploratory, added after seeing the results: P1 within neurons v6's reader flags unrelated mixed, and
    within the rest, to see whether the driver gradient only follows the reader's flag."""
    out = {}
    for name, flag in (("reader flags unrelated mixed", True), ("reader doesn't flag", False)):
        group = [r for r in rows if r["unrelated_mixed"] == flag]
        q = np.array([r["quartile"] for r in group])
        out[name] = {"neurons_by_quartile": [int(np.sum(q == k)) for k in range(4)],
                     "least_pair_unrelated_by_quartile": [round(float(np.mean([r["mlp"]["least_unrelated"] for r in group if r["quartile"] == k])), 3)
                                                          for k in range(4)],
                     "P1": top_minus_rest(q, column(group, "mlp", "least_unrelated"), rng_for(201 + flag))}
    return out


def split_half_reliability(candidates: list[dict]) -> dict:
    """Exploratory, added after seeing the results: each facet's examples split at random into two halves (seed
    [SEED, 23, 300]); the overlap of the halves' top-100 MLP drivers through the facet's own neuron. Low
    reliability means a noisy driver signature, which could make two facets look driver-unrelated."""
    data = np.load(RECORDING_PATH)
    n_facet = int(data["n_facet_examples"])
    row = {tuple(e): i for i, e in enumerate(data["examples"][:n_facet])}
    neurons = data["neurons"].tolist()
    acts, sigma = data["acts"].reshape(len(data["sigma"]), -1), data["sigma"]
    coefficients = data["coefficients"].reshape(acts.shape[1], -1)
    background = (acts[n_facet:].astype(np.float32) / sigma[n_facet:, None]).mean(axis=0)
    rng = np.random.default_rng([SEED, 23, 300])
    reliability = {}
    for c in candidates:
        k = neurons.index(c["neuron"])
        for f, facet in enumerate(c["facets"]):
            rows = rng.permutation([row[(w, p)] for w, p, _ in facet])
            halves = [np.sort(rows[:len(rows) // 2]), np.sort(rows[len(rows) // 2:])]
            sets = [drivers(((acts[h].astype(np.float32) / sigma[h, None]).mean(axis=0) - background) * coefficients[:, k], TOP_UNITS)
                    for h in halves]
            reliability[c["neuron"], f] = overlap(*sets, TOP_UNITS)
    return reliability


def reliability_check(rows: list[dict], reliability: dict) -> dict:
    """Exploratory: split-half reliability by quartile, and S2 (the margin) within terciles of the least pair's
    mean reliability, combined as in C1."""
    least = np.array([np.mean([reliability[r["neuron"], int(f)] for f in r["mlp"]["least_pair"].split("-")]) for r in rows])
    q = np.array([r["quartile"] for r in rows])
    strata = np.digitize(least, np.quantile(least, [1 / 3, 2 / 3]))
    stratified = top_minus_rest(q, column(rows, "mlp", "margin"), rng_for(301), strata)
    return {"median_facet_reliability_by_quartile": [round(float(np.median([reliability[r["neuron"], f] for r in rows if r["quartile"] == k
                                                                           for f in range(r["n_facets"])])), 3) for k in range(4)],
            "median_least_pair_reliability_by_quartile": [round(float(np.median(least[q == k])), 3) for k in range(4)],
            "margin_within_reliability_strata": {**stratified, "strata_by_quartile": strata_table(q, strata)}}


def source_shares(final: dict, rows: list[dict]) -> dict:
    """Exploratory: median share of a facet's rise over background carried directly by upstream MLP neurons and by
    attention heads (analyze_v4.step_summary's shares), by quartile."""
    return {part: [round(float(np.median([final["shares"][r["neuron"], f][part] for r in rows if r["quartile"] == k
                                          for f in range(r["n_facets"])])), 3) for k in range(4)]
            for part in ("mlp_neurons", "attention_heads")}


def reader_margin_check(rows: list[dict]) -> dict:
    """Exploratory, added after review: the reader's continuous measure (v6's least-related centroid similarity
    minus its unrelated level; higher means more related), its Spearman with the driver margin, and the primary
    measure by quartile within terciles of it, combined as in C1."""
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"][f"{MODEL} {DEPTH}"]
    reader = np.array([v6[str(r["neuron"])]["least_related_similarity"] - v6[str(r["neuron"])]["unrelated_level"] for r in rows])
    q = np.array([r["quartile"] for r in rows])
    least = column(rows, "mlp", "least_unrelated")
    strata = np.digitize(reader, np.quantile(reader, [1 / 3, 2 / 3]))
    return {"spearman_reader_margin_driver_margin": round(float(spearmanr(reader, column(rows, "mlp", "margin")).statistic), 3),
            "median_reader_margin_by_quartile": [round(float(np.median(reader[q == k])), 3) for k in range(4)],
            "by_tercile": {str(t): {"top_rate": round(float(least[(strata == t) & (q == 3)].mean()), 3), "top_n": int(np.sum((strata == t) & (q == 3))),
                                    "rest_rate": round(float(least[(strata == t) & (q < 3)].mean()), 3), "rest_n": int(np.sum((strata == t) & (q < 3)))}
                           for t in range(3)},
            "P1_within_reader_margin_terciles": top_minus_rest(q, least, rng_for(401), strata)}


def token_check(candidates: list[dict], rows: list[dict], corpus: np.ndarray) -> dict:
    """Exploratory, added after review (v13's token confound): how often the least pair's two facets share an
    activating token, and the primary measure by quartile among least pairs that share none."""
    tokens = {(c["neuron"], f): {int(corpus[w, p]) for w, p, _ in facet} for c in candidates for f, facet in enumerate(c["facets"])}
    shared = np.array([bool(tokens[r["neuron"], int(a)] & tokens[r["neuron"], int(b)])
                       for r in rows for a, b in [r["mlp"]["least_pair"].split("-")]])
    q = np.array([r["quartile"] for r in rows])
    least = column(rows, "mlp", "least_unrelated")
    return {"least_pair_shares_a_token_by_quartile": [round(float(shared[q == k].mean()), 3) for k in range(4)],
            "least_pair_unrelated_without_shared_token_by_quartile": [round(float(least[(q == k) & ~shared].mean()), 3) for k in range(4)]}


def threshold_ties(rows: list[dict]) -> dict:
    """Exploratory, added after review: least pairs whose overlap equals the threshold (unrelated under v2's rule)."""
    return {kind: [sum(r[kind]["least_overlap"] == r[kind]["threshold"] for r in rows if r["quartile"] == k) for k in range(4)]
            for kind in KINDS}


def background_overlap(candidates: list[dict]) -> dict:
    """How many of v13's background positions, and of their windows, hold v23 facet examples."""
    examples = np.load(RECORDING_PATH)["examples"]
    n_facet = int(np.load(RECORDING_PATH)["n_facet_examples"])
    facet_examples = {tuple(e) for e in examples[:n_facet].tolist()}
    facet_windows = {w for w, _ in facet_examples}
    background = [tuple(e) for e in examples[n_facet:].tolist()]
    return {"positions": len(background), "positions_that_are_facet_examples": sum(e in facet_examples for e in background),
            "windows_holding_facet_examples": sum(w in facet_windows for w, _ in background)}


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
    final = final_summary(candidates)
    rows = neuron_rows(candidates, final)
    summary = {"neurons_by_quartile": eligible_counts() | {"selected": [sum(r["quartile"] == k for r in rows) for k in range(4)]},
               "facets": len([s for r in rows for s in r["facet_sizes"]]), "pairs": sum(len(r["mlp"]["overlaps"]) for r in rows),
               "facet_count_distribution": {str(k): v for k, v in sorted(Counter(r["n_facets"] for r in rows).items())},
               "reconstruction_error": final["reconstruction_error"], "background": background_overlap(candidates),
               "pre_registered": pre_registered(rows), "by_quartile": by_quartile(rows),
               "share_of_pairs_unrelated": {kind: round(float(np.mean([o <= r[kind]["threshold"] for r in rows
                                                                       for o in r[kind]["overlaps"].values()])), 3) for kind in KINDS},
               "v13_check": v13_check(rows),
               "exploratory_after_results": {"reader_split": reader_split(rows),
                                             "split_half_reliability": reliability_check(rows, split_half_reliability(candidates)),
                                             "source_shares_by_quartile": source_shares(final, rows)},
               "exploratory_after_review": {"reader_margin": reader_margin_check(rows),
                                            "shared_tokens": token_check(candidates, rows, load_corpus()),
                                            "least_pairs_tied_with_threshold_by_quartile": threshold_ties(rows)}}
    (RESULTS_DIR / "classification_v23.json").write_text(json.dumps({"summary": summary, "neurons": rounded(rows)}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
