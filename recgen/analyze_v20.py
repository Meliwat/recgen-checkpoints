"""Apply the pre-registered v20 rules (recgen/rubric_v20.py): at matched selectivity, from the same ranks of
each neuron's own window maxima, do strongly firing neurons' examples still look more coherent and less mixed
than weakly firing neurons'? And how does that compare with v17's band at a fixed fraction of the peak?

Reads cached data only: v5's window maxima, v6's recordings and results, v20's recordings (data/v20/), v12's,
v17's and v20's Haiku answers, and v17's per-neuron band readings. Writes results/classification_v20.json.
"""

import json

import numpy as np

from recgen.analyze_v12 import haiku_rows
from recgen.analyze_v17 import by_model, flag_given_groups, haiku_checks, label_shares, network_rows
from recgen.band_v17 import band_examples
from recgen.band_v20 import rank_band_examples, ranked
from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.read_v12 import ANSWERS_PATH as V12_ANSWERS_PATH
from recgen.read_v17 import ANSWERS_PATH as V17_ANSWERS_PATH
from recgen.read_v17 import band_units
from recgen.read_v20 import ANSWERS_PATH, rank_units
from recgen.record_v20 import SECOND_BAND, second_band_path, v20_path
from recgen.rubric_v5 import MODELS
from recgen.rubric_v6 import token_purity
from recgen.rubric_v7 import UNRELATED_JOBS
from recgen.rubric_v12 import h2
from recgen.rubric_v17 import BAND as V17_BAND
from recgen.rubric_v17 import ols_coefficients
from recgen.rubric_v20 import (
    DEPTH, MIXED_SIGN, PRIMARY_BAND, RANKS, READINGS, against_v17, answer, rng_for, t1, t2,
)
from recgen.sweep import sweep_path

NETWORK_MEASURES = ("flag", "n_facets", "coherence")
SAMPLES = ("top", "band", "band2", "v17", "reference")
INCLUDED = {PRIMARY_BAND: "included", SECOND_BAND: "band2_included"}  # network row fields


def in_v17_band(fractions: np.ndarray) -> np.ndarray:
    return (fractions >= V17_BAND[0]) & (fractions < V17_BAND[1])


def v17_band_ranks(model: str) -> np.ndarray:
    """The rank (1 is the largest) of each of v17's band examples among its neuron's window maxima [S, 50], with
    v20's ranking; 0 for neurons without a v17 band sample."""
    band = band_examples(model)
    window_max = np.load(sweep_path(model, DEPTH))["max"]
    ranks = np.zeros(band["keys"].shape[:2], dtype=np.int32)
    for s, n in enumerate(band["neurons"]):
        if band["included"][s]:
            rank_of = np.empty(len(window_max), dtype=np.int32)
            rank_of[ranked(window_max[:, n].astype(np.float32))] = np.arange(1, len(window_max) + 1)
            ranks[s] = rank_of[band["keys"][s][:, 0]]
    return ranks


def network_rows_v20(model: str, published: dict, v17_neurons: dict, corpus: np.ndarray) -> tuple[list[dict], dict]:
    """Every sampled neuron at DEPTH: v17's network_rows on the rank band (top 50, "band", reference, covariates,
    with v6's published top-50 values checked), joined with the second band's readings, v17's band readings and
    each sample's fractions of peak and ranks."""
    band = rank_band_examples(model, PRIMARY_BAND)
    rows, facts = network_rows(model, published, corpus, v20_path(model), band)
    recorded = np.load(v20_path(model))
    assert np.array_equal(recorded["fractions"], band["fractions"], equal_nan=True)
    second = np.load(second_band_path(model))
    assert np.array_equal(second["neurons"], band["neurons"])
    assert np.array_equal(second["keys"], rank_band_examples(model, SECOND_BAND)["keys"])
    v17_ranks = v17_band_ranks(model)
    for s, row in enumerate(rows):
        row["band_fractions"] = band["fractions"][s]
        row["band2_included"] = bool(second["included"][s])
        if row["band2_included"]:
            row["band2_fractions"] = second["fractions"][s]
            for measure in NETWORK_MEASURES:
                row[f"band2_{measure}"] = float(second[measure][s])
        v17 = v17_neurons.get(str(row["neuron"]))
        row["v17_included"] = v17 is not None
        if v17 is not None:
            row["v17_flag"], row["v17_n_facets"], row["v17_coherence"] = (float(x) for x in v17)
            row["v17_ranks"] = v17_ranks[s]
    facts["band2"] = {"activation_error": float(second["activation_error"]),
                      "median_band_activation": float(second["median_band_activation"]),
                      "excluded": int((~second["included"]).sum())}
    facts["seconds"] = round(float(second["seconds"]))
    return rows, facts


def haiku_rank_rows(v12_answers: dict, v17_answers: dict, v20_answers: dict, v6: dict, corpus: np.ndarray) -> list[dict]:
    """v12's fresh-neuron rows (covariates, quartile, top-50 flag) with a valid label for the rank band, joined with
    the second band's and v17's band readings where those have a valid label (None otherwise)."""
    by_neuron = {(r["model"], r["neuron"]): r for r in haiku_rows(v12_answers, v6, corpus)}

    def flag(label):
        return None if label is None else float(label == UNRELATED_JOBS)

    rows = []
    for model in MODELS:
        second = {u["neuron_id"]: u for u in rank_units(model, SECOND_BAND)}
        v17 = {u["neuron_id"]: u for u in band_units(model)}
        for u in rank_units(model, PRIMARY_BAND):
            label = v20_answers[PRIMARY_BAND][u["id"]]["label"]
            top = by_neuron.get((model, u["neuron_id"]))
            if label is None or top is None:
                continue
            second_label = v20_answers[SECOND_BAND][second[u["neuron_id"]]["id"]]["label"] if u["neuron_id"] in second else None
            v17_label = v17_answers["band"][v17[u["neuron_id"]]["id"]]["label"] if u["neuron_id"] in v17 else None
            rows.append({**top, "top_flag": top["haiku"], "band_flag": flag(label), "band2_flag": flag(second_label),
                         "v17_flag": flag(v17_label), "band_token_purity": token_purity([int(corpus[w, p]) for w, p in u["keys"]]),
                         "top_label": v12_answers[u["top"]["id"]]["label"], "band_label": label, "band2_label": second_label})
    return rows


def with_value(rows: list[dict], field: str) -> list[dict]:
    return [r for r in rows if r.get(field) is not None]


def band_tests(rows: list[dict], measure: str, band: str, purpose: int) -> dict:
    """T1 and T2 for one band, reader and measure, on rows with both a band and a top-50 value, and the verdict's
    pre-registered reading."""
    rows = with_value(rows, f"{band}_{measure}")
    sign = MIXED_SIGN[measure]
    pairs = by_model(rows, f"{band}_{measure}", f"top_{measure}")
    groups = by_model(rows, "log_peak", "quartile", f"{band}_{measure}", f"top_{measure}")
    out = {"n": len(rows), "t1": t1(pairs, sign, rng_for(MODELS[0], purpose)), "t2": t2(groups, sign, rng_for(MODELS[0], purpose + 1))}
    out["reading"] = READINGS[out["t2"]["verdict"]]
    return out


def by_quartile(rows: list[dict], measure: str) -> dict:
    """Each sample's mean by peak quartile, with its number of neurons."""
    out = {}
    for sample in SAMPLES:
        present = with_value(rows, f"{sample}_{measure}")
        if present:
            out[sample] = {"n": len(present), "mean": [float(np.mean([r[f"{sample}_{measure}"] for r in present if r["quartile"] == q]))
                                                       for q in range(4)]}
    return out


def comparison(rows: list[dict], measure: str, purpose: int) -> dict:
    """Descriptive: v12's H1 statistic in v17's band and in the rank band, and the paired difference, on neurons
    with both."""
    rows = with_value(rows, f"v17_{measure}")
    groups = by_model(rows, "log_peak", "quartile", f"band_{measure}", f"v17_{measure}")
    return {"n": len(rows), **against_v17(groups, MIXED_SIGN[measure], rng_for(MODELS[0], purpose))}


def pooled_by_quartile(rows: list[dict], field: str, statistic) -> list[list[float]]:
    """By peak quartile, the range over models of statistic over the pooled per-example values in field."""
    values = [[float(statistic(np.concatenate([r[field] for r in rows if r["model"] == model and r["quartile"] == q])))
               for model in MODELS] for q in range(4)]
    return [[round(min(v), 3), round(max(v), 3)] for v in values]


def selectivity(rows: list[dict]) -> dict:
    """Descriptive: where each band's examples sit in their neurons' range, by peak quartile (range over models)."""
    out = {}
    for band in RANKS:
        present = [r for r in rows if f"{band}_fractions" in r]
        out[band] = {"median_fraction_of_peak": pooled_by_quartile(present, f"{band}_fractions", np.median),
                     "share_in_v17_band": pooled_by_quartile(present, f"{band}_fractions", lambda f: in_v17_band(f).mean())}
    out["v17_band"] = {"median_rank": pooled_by_quartile(with_value(rows, "v17_ranks"), "v17_ranks", np.median)}
    return out


def secondary(rows: list[dict], measure: str, purpose: int) -> dict:
    """S1 (v12's H2 on rank-band flags) or S2 (least squares on oriented rank-band coherence), with the rank-band
    examples' token purity in place of the top 50's."""
    regression_rows = [{"model": r["model"], "log_peak": r["log_peak"], "peakedness": r["peakedness"],
                        "density": r["density"], "token_purity": r["band_token_purity"],
                        "flag": MIXED_SIGN[measure] * r[f"band_{measure}"]} for r in rows]
    rng = rng_for(MODELS[0], purpose)
    return h2(regression_rows, rng) if measure == "flag" else ols_coefficients(regression_rows, rng)


def facets_by_quartile(rows: list[dict]) -> dict:
    """Exploratory, added after seeing the results: by peak quartile, the network reader's shares of neurons with no
    facet and with 2+ facets in each sample. The flag needs 2+ facets, so diffuse examples can lower it (v17's caveat)."""
    out = {}
    for sample in ("top", "band", "band2", "v17"):
        present = with_value(rows, f"{sample}_n_facets")
        counts = [np.array([r[f"{sample}_n_facets"] for r in present if r["quartile"] == q]) for q in range(4)]
        out[sample] = {"no_facet": [float(np.mean(c == 0)) for c in counts], "two_plus": [float(np.mean(c >= 2)) for c in counts]}
    return out


def main() -> None:
    corpus = load_corpus()
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"]
    v17_neurons = json.loads((RESULTS_DIR / "classification_v17.json").read_text())["neurons"]
    v12_answers = json.loads(V12_ANSWERS_PATH.read_text())
    v17_answers = json.loads(V17_ANSWERS_PATH.read_text())
    v20_answers = json.loads(ANSWERS_PATH.read_text())
    network, recording = [], {}
    for model in MODELS:
        rows, recording[model] = network_rows_v20(model, v6[f"{model} {DEPTH}"], v17_neurons[model], corpus)
        network += rows
        print(f"pythia-{model}: {sum(r['included'] for r in rows)} of {len(rows)} neurons in the rank band, "
              f"{sum(r['band2_included'] for r in rows)} in the second", flush=True)
    included = [r for r in network if r["included"]]
    haiku = haiku_rank_rows(v12_answers, v17_answers, v20_answers, v6, corpus)
    readers = {("network", "flag"): included, ("network", "coherence"): included, ("haiku", "flag"): haiku}
    results = {"tests": {}, "verdicts": {}, "readings": {}, "answer": {}, "by_quartile": {}, "against_v17": {}}
    purposes = iter(range(0, 1000, 10))  # each test gets its own bootstrap streams
    for (reader, measure), rows in readers.items():
        key = f"{reader}, {measure}"
        results["by_quartile"][key] = by_quartile(rows, measure)
        results["against_v17"][key] = comparison(rows, measure, next(purposes))
        for band in RANKS:
            tests = band_tests(rows, measure, band, next(purposes))
            results["tests"][f"{key}, {band}"] = tests
            results["verdicts"][f"{key}, {band}"] = tests["t2"]["verdict"]
            results["readings"][f"{key}, {band}"] = tests["reading"]
    for band in RANKS:
        results["answer"][band] = answer(results["verdicts"][f"network, coherence, {band}"],
                                         results["verdicts"][f"haiku, flag, {band}"])
    checks = haiku_checks(haiku, v12_answers, v20_answers)
    second_labels = [r["band2_label"] for r in haiku if r["band2_label"] is not None]
    checks["label_shares"]["band2"] = label_shares(second_labels)
    checks["readings"] += len(v20_answers[SECOND_BAND])
    checks["without_label"] += sum(a["label"] is None for a in v20_answers[SECOND_BAND].values())
    results.update({
        "secondary": {"network, flag (S1)": secondary(included, "flag", 200),
                      "network, coherence (S2)": secondary(included, "coherence", 201),
                      "haiku, flag (S1)": secondary(haiku, "flag", 202)},
        "selectivity": selectivity(included),
        "exclusions": {"network": {band: {model: [sum(1 for r in network if r["model"] == model and r["quartile"] == q
                                                      and not r[INCLUDED[band]]) for q in range(4)] for model in MODELS}
                                   for band in RANKS},
                       "network_without_v17_band": sum(not r["v17_included"] for r in included),
                       "haiku_fresh_neurons_without_rank_band_sample": {band: 600 - sum(len(rank_units(m, band)) for m in MODELS)
                                                                         for band in RANKS},
                       "haiku_without_valid_label": {band: sum(a["label"] is None for a in v20_answers[band].values())
                                                     for band in RANKS},
                       "haiku_neurons": {sample: len(with_value(haiku, f"{sample}_flag")) for sample in ("band", "band2", "v17")}},
        "recording": {"by_model": recording,
                      "largest_activation_error": {"band": max(r["activation_error"] for r in recording.values()),
                                                   "band2": max(r["band2"]["activation_error"] for r in recording.values())},
                      "seconds": sum(r["seconds"] for r in recording.values()),
                      "stored_bytes": sum(path(m).stat().st_size for m in MODELS for path in (v20_path, second_band_path))},
        "haiku_checks": checks,
        "exploratory_network_facets_by_quartile": facets_by_quartile(included),
        "exploratory_network_flag_given_two_plus_facets": {
            sample: flag_given_groups(with_value(included, f"{sample}_n_facets"), sample, 300 + k)
            for k, sample in enumerate(("top", "band", "band2"))},
        "neurons": {model: {str(r["neuron"]): [r["band_flag"], r["band_n_facets"], round(r["band_coherence"], 2)]
                            + ([r["band2_flag"], int(r["band2_n_facets"]), round(r["band2_coherence"], 2)] if r["band2_included"] else [])
                            for r in included if r["model"] == model} for model in MODELS},
    })
    (RESULTS_DIR / "classification_v20.json").write_text(json.dumps(results, indent=1))
    report(results)


def report(results: dict) -> None:
    def interval(x):
        return [round(v, 3) for v in x]

    for key, r in results["tests"].items():
        a, b = r["t1"], r["t2"]
        print(f"{key} (n={r['n']}): T1 band minus top {a['band_minus_top']:+.3f} {interval(a['interval'])}: {a['reading']}")
        print(f"  T2 band {b['band']['top_minus_rest']:+.3f} {interval(b['band']['interval'])} ({b['band']['models_negative']}/6, "
              f"holds {b['band']['holds']}); top {b['top']['top_minus_rest']:+.3f} {interval(b['top']['interval'])}; "
              f"band minus top {b['band_minus_top']['difference']:+.3f} {interval(b['band_minus_top']['interval'])} "
              f"-> {b['verdict']} ({r['reading']})")
    print("answer:", results["answer"])
    for key, r in results["against_v17"].items():
        print(f"against v17, {key} (n={r['n']}): rank band {r['rank_band']['top_minus_rest']:+.3f} "
              f"{interval(r['rank_band']['interval'])}, v17 band {r['v17_band']['top_minus_rest']:+.3f} "
              f"{interval(r['v17_band']['interval'])}, rank minus v17 {r['rank_minus_v17']['difference']:+.3f} "
              f"{interval(r['rank_minus_v17']['interval'])}")
    for key, r in results["by_quartile"].items():
        print(f"{key} by quartile: " + "; ".join(f"{s} {[round(x, 3) for x in v['mean']]} (n={v['n']})" for s, v in r.items()))
    for key, r in results["secondary"].items():
        print(f"{key}: log peak {r['log_peak']['coefficient']:+.3f} {interval(r['log_peak']['interval'])}, "
              f"density {r['density']['coefficient']:+.3f} {interval(r['density']['interval'])}, holds {r['holds']}")
    for sample, r in results["exploratory_network_flag_given_two_plus_facets"].items():
        print(f"exploratory, network flag among neurons with 2+ {sample} facets (n={r['n']}): by quartile "
              f"{[round(x, 3) for x in r['by_quartile']]}, top minus rest {r['h1']['top_minus_rest']:+.3f} "
              f"{interval(r['h1']['interval'])}, {r['h1']['models_negative']}/6 negative")
    print(json.dumps({k: results[k] for k in ("selectivity", "exclusions", "recording", "haiku_checks",
                                              "exploratory_network_facets_by_quartile")}, indent=1))


if __name__ == "__main__":
    main()
