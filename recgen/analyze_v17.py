"""Apply the pre-registered v17 rules (recgen/rubric_v17.py): below the top 50, at a fixed fraction of
each neuron's own peak, do weak neurons still look more mixed, and do a neuron's band examples look
more mixed than its top 50?

Reads cached data only: v5's window maxima, v6's recordings and results, v17's band recordings
(data/v17/), v12's and v17's Haiku answers. Writes results/classification_v17.json.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recgen.analyze_v6 import PRIMARY, Layer, detect, levels
from recgen.analyze_v12 import haiku_rows, layer_covariates
from recgen.band_v17 import band_examples
from recgen.config import N_WINDOWS, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.read_v12 import ANSWERS_PATH as V12_ANSWERS_PATH
from recgen.read_v17 import ANSWERS_PATH, band_units
from recgen.record_v17 import v17_path
from recgen.rubric_v5 import MODELS
from recgen.rubric_v6 import MERGE_PERCENTILE, cohen_kappa, embed, pair_similarities, percentile_rank, token_purity
from recgen.rubric_v7 import UNRELATED_JOBS
from recgen.rubric_v12 import h1, h2, quartiles
from recgen.rubric_v17 import BAND_SIZE, DEPTH, MIXED_SIGN, ols_coefficients, reference_rng, rng_for, t1, t2
from recgen.sweep_v6 import v6_path

THREADS = 8
SAMPLES = ("top", "band", "reference")


def read_examples(units: np.ndarray, null: np.ndarray) -> dict:
    """v6's detector at its pre-registered settings, and v6's top_pair_percentile, on embedded examples."""
    lv = levels(null)
    result = detect(units, lv["merge"][MERGE_PERCENTILE], lv["unrelated"])
    return {"flag": float(result["unrelated_mixed"]), "n_facets": result["n_facets"],
            "coherence": percentile_rank(float(np.median(pair_similarities(units))), null)}


def neuron_readings(layer: Layer, band_z: np.ndarray, band_rows: np.ndarray, reference_rows: np.ndarray, s: int) -> dict:
    """The network reader on sampled neuron s's top 50 (v6's examples), its band examples (None if it has
    no band sample) and its random-position reference, embedded as Layer.neuron embeds them."""
    mean, _, directions = layer.embeddings[PRIMARY]
    direction = directions[[s]]
    null = layer.nulls[PRIMARY].similarities(direction)
    return {"top": read_examples(embed(layer.z[layer.rows[s]], mean, direction), null),
            "band": read_examples(embed(band_z[band_rows], mean, direction), null) if band_rows[0] >= 0 else None,
            "reference": read_examples(embed(layer.z[reference_rows], mean, direction), null)}


def network_rows(model: str, published: dict, corpus: np.ndarray) -> tuple[list[dict], dict]:
    """Every sampled neuron at DEPTH with v12's covariates and the network reader on its three samples,
    checked against v6's published top-50 results; and the layer's recording facts."""
    layer = Layer(model, DEPTH)
    recorded = np.load(v17_path(model))
    band = band_examples(model)
    assert np.array_equal(recorded["neurons"], layer.neurons) and np.array_equal(band["neurons"], layer.neurons)
    band_z, band_rows = recorded["z"], recorded["band_rows"]
    assert np.array_equal(recorded["keys"][band_rows[band["included"]]], band["keys"][band["included"]])
    random_rows = np.load(v6_path(model, DEPTH))["random_rows"]
    references = [reference_rng(model, int(n)).choice(random_rows, size=BAND_SIZE, replace=False) for n in layer.neurons]
    with ThreadPoolExecutor(THREADS) as pool:
        readings = list(pool.map(lambda s: neuron_readings(layer, band_z, band_rows[s], references[s], s),
                                 range(len(layer.neurons))))
    ids = [int(n) for n in layer.neurons]
    covariates = layer_covariates(model, DEPTH, ids, corpus)
    quartile = quartiles(np.array([covariates[n]["peak"] for n in ids]))
    rows = []
    for s, (n, r) in enumerate(zip(ids, readings)):
        v6 = published[str(n)]
        assert r["top"]["flag"] == v6["unrelated_mixed"] and r["top"]["n_facets"] == v6["n_facets"]
        assert abs(r["top"]["coherence"] - v6["top_pair_percentile"]) < 1e-3
        row = {"model": model, "neuron": n, "s": s, "quartile": int(quartile[s]), **covariates[n],
               "band_count": int(band["counts"][s]), "included": bool(band["included"][s])}
        for sample in SAMPLES:
            for measure, value in (r[sample] or {}).items():
                row[f"{sample}_{measure}"] = value
        if row["included"]:
            row["band_token_purity"] = token_purity([int(corpus[w, p]) for w, p in band["keys"][s]])
        rows.append(row)
    facts = {"activation_error": float(recorded["activation_error"]),
             "median_band_activation": float(recorded["median_band_activation"]),
             "positions": int(len(recorded["keys"])), "seconds": round(float(recorded["seconds"]))}
    return rows, facts


def haiku_band_rows(v12_answers: dict, v17_answers: dict, v6: dict, corpus: np.ndarray) -> list[dict]:
    """v12's fresh-neuron rows (covariates, quartile, top-50 flag) joined with v17's band readings, for
    neurons with a valid label in both."""
    by_neuron = {(r["model"], r["neuron"]): r for r in haiku_rows(v12_answers, v6, corpus)}
    rows = []
    for model in MODELS:
        for u in band_units(model):
            label = v17_answers["band"][u["id"]]["label"]
            top = by_neuron.get((model, u["neuron_id"]))
            if label is None or top is None:
                continue
            rows.append({**top, "top_flag": top["haiku"], "band_flag": float(label == UNRELATED_JOBS),
                         "band_token_purity": token_purity([int(corpus[w, p]) for w, p in u["keys"]]),
                         "top_label": v12_answers[u["top"]["id"]]["label"], "band_label": label})
    return rows


def by_model(rows: list[dict], *fields: str) -> list[tuple[np.ndarray, ...]]:
    return [tuple(np.array([r[f] for r in rows if r["model"] == model], float) for f in fields) for model in MODELS]


def reader_tests(rows: list[dict], measure: str, purpose: int) -> dict:
    """T1 and T2 for one reader and measure on rows with both a band and a top-50 value."""
    sign = MIXED_SIGN[measure]
    pairs = by_model(rows, f"band_{measure}", f"top_{measure}")
    groups = by_model(rows, "log_peak", "quartile", f"band_{measure}", f"top_{measure}")
    by_quartile = {sample: [float(np.mean([r[f"{sample}_{measure}"] for r in rows if r["quartile"] == q])) for q in range(4)]
                   for sample in SAMPLES if f"{sample}_{measure}" in rows[0]}
    return {"n": len(rows), "by_quartile": by_quartile, "t1": t1(pairs, sign, rng_for(MODELS[0], purpose)),
            "t2": t2(groups, sign, rng_for(MODELS[0], purpose + 1))}


def secondary(rows: list[dict], measure: str, purpose: int) -> dict:
    """S1 (v12's H2 on band flags) or S2 (least squares on oriented band coherence), with the band
    examples' token purity in place of the top 50's."""
    regression_rows = [{"model": r["model"], "log_peak": r["log_peak"], "peakedness": r["peakedness"],
                        "density": r["density"], "token_purity": r["band_token_purity"],
                        "flag": MIXED_SIGN[measure] * r[f"band_{measure}"]} for r in rows]
    rng = rng_for(MODELS[0], purpose)
    return h2(regression_rows, rng) if measure == "flag" else ols_coefficients(regression_rows, rng)


def agreement(first: list[str], second: list[str]) -> dict:
    flags = [np.array([label == UNRELATED_JOBS for label in labels]) for labels in (first, second)]
    return {"n": len(first), "same_label": float(np.mean([a == b for a, b in zip(first, second)])),
            "same_flag": float(np.mean(flags[0] == flags[1])), "kappa": cohen_kappa(*flags),
            "flag_rates": [float(f.mean()) for f in flags]}


def label_shares(labels: list[str]) -> dict:
    return {label: float(np.mean([x == label for x in labels])) for label in ("A", "B", "C")}


def haiku_checks(rows: list[dict], v12_answers: dict, v17_answers: dict) -> dict:
    """Label shares, the band retest, and the drift check (v12's top-50 prompts read again)."""
    retest = [(v17_answers["band"][i]["label"], a["label"]) for i, a in v17_answers["retest"].items()]
    reread = [(v12_answers[i]["label"], a["label"]) for i, a in v17_answers["top_reread"].items()]
    retest, reread = ([p for p in pairs if None not in p] for pairs in (retest, reread))
    return {"label_shares": {"top": label_shares([r["top_label"] for r in rows]),
                             "band": label_shares([r["band_label"] for r in rows])},
            "band_retest": agreement(*zip(*retest)),
            "top_reread_against_v12": agreement(*zip(*reread)),
            "cost_usd": v17_answers["total_cost_usd"],
            "readings": sum(len(v17_answers[k]) for k in ("band", "retest", "top_reread")),
            "without_label": sum(a["label"] is None for k in ("band", "retest", "top_reread") for a in v17_answers[k].values())}


def facet_counts(rows: list[dict]) -> dict:
    """Exploratory, added after a first look at 70M's band flags: by peak quartile, the network reader's mean
    facet count and the shares of neurons with no facet and with 2+ facets, for each sample. The flag needs
    2+ facets, so diffuse examples can lower it (the pre-registered caveat)."""
    out = {}
    for sample in SAMPLES:
        counts = [np.array([r[f"{sample}_n_facets"] for r in rows if r["quartile"] == q]) for q in range(4)]
        out[sample] = {"mean": [float(c.mean()) for c in counts], "no_facet": [float(np.mean(c == 0)) for c in counts],
                       "two_plus": [float(np.mean(c >= 2)) for c in counts]}
    return out


def flag_given_groups(rows: list[dict], sample: str, purpose: int) -> dict:
    """Exploratory, added after seeing the results: v12's H1 on the network flag among neurons whose sample
    has 2+ facets, the only ones that can be flagged. The subsets differ between samples, so this isn't
    paired, and conditioning on the facet count selects neurons differently by quartile."""
    subset = [r for r in rows if r[f"{sample}_n_facets"] >= 2]
    return {"n": len(subset),
            "by_quartile": [float(np.mean([r[f"{sample}_flag"] for r in subset if r["quartile"] == q])) for q in range(4)],
            "h1": h1(by_model(subset, "log_peak", "quartile", f"{sample}_flag"), rng_for(MODELS[0], purpose))}


def model_range(rows: list[dict], value) -> list[list[float]]:
    """By peak quartile, the range over models of the median of value(row)."""
    medians = [[float(np.median([value(r) for r in rows if r["model"] == model and r["quartile"] == q])) for model in MODELS]
               for q in range(4)]
    return [[round(min(m), 3), round(max(m), 3)] for m in medians]


def main() -> None:
    corpus = load_corpus()
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"]
    v12_answers = json.loads(V12_ANSWERS_PATH.read_text())
    v17_answers = json.loads(ANSWERS_PATH.read_text())
    network, recording = [], {}
    for model in MODELS:
        rows, recording[model] = network_rows(model, v6[f"{model} {DEPTH}"], corpus)
        network += rows
        print(f"pythia-{model}: {sum(r['included'] for r in rows)} of {len(rows)} neurons with a band sample", flush=True)
    included = [r for r in network if r["included"]]
    haiku = haiku_band_rows(v12_answers, v17_answers, v6, corpus)
    results = {
        "network": {measure: reader_tests(included, measure, 10 * k) for k, measure in enumerate(MIXED_SIGN)},
        "haiku": {"flag": reader_tests(haiku, "flag", 30)},
        "verdicts": {},
        "secondary": {"network, flag (S1)": secondary(included, "flag", 40),
                      "network, coherence (S2)": secondary(included, "coherence", 41),
                      "haiku, flag (S1)": secondary(haiku, "flag", 42)},
        "exclusions": {"network": {model: [sum(1 for r in network if r["model"] == model and not r["included"] and r["quartile"] == q)
                                           for q in range(4)] for model in MODELS},
                       "haiku_fresh_neurons_without_band_sample": 600 - sum(len(band_units(m)) for m in MODELS),
                       "haiku_without_valid_label": sum(len(band_units(m)) for m in MODELS) - len(haiku)},
        "band_share_of_windows": model_range(network, lambda r: r["band_count"] / N_WINDOWS),
        "recording": {"by_model": recording,
                      "largest_activation_error": max(r["activation_error"] for r in recording.values()),
                      "seconds": sum(r["seconds"] for r in recording.values())},
        "haiku_checks": haiku_checks(haiku, v12_answers, v17_answers),
        "exploratory_network_facets_by_quartile": facet_counts(included),
        "exploratory_network_flag_given_two_plus_facets": {sample: flag_given_groups(included, sample, 50 + k)
                                                           for k, sample in enumerate(("top", "band"))},
        "neurons": {model: {str(r["neuron"]): [r["band_flag"], r["band_n_facets"], round(r["band_coherence"], 2)]
                            for r in included if r["model"] == model} for model in MODELS},
    }
    for reader in ("network", "haiku"):
        for measure, r in results[reader].items():
            results["verdicts"][f"{reader}, {measure}"] = r["t2"]["verdict"]
    (RESULTS_DIR / "classification_v17.json").write_text(json.dumps(results, indent=1))
    for reader in ("network", "haiku"):
        for measure, r in results[reader].items():
            a, b = r["t1"], r["t2"]
            print(f"{reader} {measure} (n={r['n']}): by quartile " + "; ".join(
                f"{k} {[round(x, 3) for x in v]}" for k, v in r["by_quartile"].items()))
            print(f"  T1 band minus top {a['band_minus_top']:+.3f} {[round(x, 3) for x in a['interval']]}: {a['reading']}")
            print(f"  T2 band {b['band']['top_minus_rest']:+.3f} {[round(x, 3) for x in b['band']['interval']]} "
                  f"({b['band']['models_negative']}/6 negative, holds {b['band']['holds']}); top {b['top']['top_minus_rest']:+.3f} "
                  f"{[round(x, 3) for x in b['top']['interval']]}; band minus top {b['band_minus_top']['difference']:+.3f} "
                  f"{[round(x, 3) for x in b['band_minus_top']['interval']]} -> {b['verdict']}")
    for key, r in results["secondary"].items():
        print(f"{key}: log peak {r['log_peak']['coefficient']:+.3f} {[round(x, 3) for x in r['log_peak']['interval']]}, "
              f"density {r['density']['coefficient']:+.3f} {[round(x, 3) for x in r['density']['interval']]}, holds {r['holds']}")
    for sample, r in results["exploratory_network_flag_given_two_plus_facets"].items():
        print(f"exploratory, network flag among neurons with 2+ {sample} facets (n={r['n']}): "
              f"by quartile {[round(x, 3) for x in r['by_quartile']]}, top minus rest {r['h1']['top_minus_rest']:+.3f} "
              f"{[round(x, 3) for x in r['h1']['interval']]}, {r['h1']['models_negative']}/6 negative")
    print(json.dumps({k: results[k] for k in ("exclusions", "band_share_of_windows", "recording", "haiku_checks")}, indent=1))


if __name__ == "__main__":
    main()
