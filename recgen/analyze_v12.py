"""Apply the pre-registered v12 rules (recgen/rubric_v12.py): does a neuron's peak activation predict how
mixed it looks, on data the v9 finding didn't use?

Reads cached data only: v5's window maxima and flags, v6's flags, and v12's fresh Haiku answers.
The middle-layer network and MiniLM results are where the effect was found, so they're reported as
"discovery", apart from the held-out tests.
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.read_v12 import ANSWERS_PATH, fresh_units
from recgen.rubric_v5 import DEPTHS, MODELS
from recgen.rubric_v7 import UNRELATED_JOBS
from recgen.rubric_v12 import (BOOTSTRAP, COVARIATES, HAIKU_DEPTH, HELD_OUT_DEPTHS, h1, h2, neuron_stats, quartiles,
                                rng_for, standardize)
from recgen.sweep import sweep_path

UNRELATED_SIMILARITY = 0.1  # v5's cutoff for MiniLM's flag
WITH_COHERENCE = COVARIATES + ("coherence",)


def layer_covariates(model: str, depth: str, neuron_ids: list[int], corpus: np.ndarray) -> dict[int, dict]:
    data = np.load(sweep_path(model, depth))
    window_max, window_pos = data["max"], data["pos"]
    stats = {}
    for n in neuron_ids:
        column = window_max[:, n].astype(np.float32)
        top = np.argsort(-column, kind="stable")[:50]
        stats[n] = neuron_stats(column, [int(corpus[w, window_pos[w, n]]) for w in top])
    median_peak = np.median([s["peak"] for s in stats.values()])
    for s in stats.values():
        s["log_peak"] = float(np.log(s["peak"] / median_peak))
    return stats


def detector_rows(model: str, depth: str, v5: dict, v6: dict, corpus: np.ndarray) -> list[dict]:
    """Every sampled neuron of a layer with its covariates and the network and MiniLM flags."""
    network, minilm = v6[f"{model} {depth}"], v5[f"{model} {depth}"]
    ids = [int(n) for n in network]
    covariates = layer_covariates(model, depth, ids, corpus)
    quartile = dict(zip(ids, quartiles(np.array([covariates[n]["peak"] for n in ids]))))
    return [{"model": model, "neuron": n, "quartile": int(quartile[n]), **covariates[n],
             "coherence": network[str(n)]["top_pair_percentile"],
             "network": float(network[str(n)]["unrelated_mixed"]),
             "minilm": float(minilm[str(n)]["n_facets"] >= 2 and minilm[str(n)]["least_related_similarity"] < UNRELATED_SIMILARITY)}
            for n in ids]


def haiku_rows(answers: dict, v6: dict, corpus: np.ndarray) -> list[dict]:
    rows = []
    for model in MODELS:
        units = fresh_units(model)
        covariates = layer_covariates(model, HAIKU_DEPTH, [u["neuron_id"] for u in units], corpus)
        # log peak relative to all sampled neurons of the layer, as for the detectors
        layer_peaks = np.load(sweep_path(model, HAIKU_DEPTH))["max"].astype(np.float32)
        network = v6[f"{model} {HAIKU_DEPTH}"]
        sampled = [int(n) for n in network]
        median_peak = np.median(layer_peaks[:, sampled].max(axis=0))
        for u in units:
            label = answers[u["id"]]["label"]
            if label is None:
                continue
            c = dict(covariates[u["neuron_id"]])
            c["log_peak"] = float(np.log(c["peak"] / median_peak))
            rows.append({"model": model, "neuron": u["neuron_id"], "quartile": u["quartile"], **c,
                         "coherence": network[str(u["neuron_id"])]["top_pair_percentile"],
                         "haiku": float(label == UNRELATED_JOBS)})
    return rows


def reader_tests(rows: list[dict], reader: str, purpose: int) -> dict:
    groups = []
    for model in MODELS:
        r = [x for x in rows if x["model"] == model]
        groups.append((np.array([x["log_peak"] for x in r]), np.array([x["quartile"] for x in r]),
                       np.array([x[reader] for x in r])))
    by_quartile = [float(np.mean([x[reader] for x in rows if x["quartile"] == q])) for q in range(4)]
    regression_rows = [{**{k: x[k] for k in ("model", "log_peak", "peakedness", "density", "token_purity")}, "flag": x[reader]}
                       for x in rows]
    return {"n": len(rows), "by_quartile": by_quartile, "h1": h1(groups, rng_for(MODELS[0], purpose)),
            "h2": h2(regression_rows, rng_for(MODELS[0], purpose + 1))}


def with_coherence(rows: list[dict], reader: str, rng) -> dict:
    """Exploratory, added after v9's review found that the top-50 examples of strong neurons are more
    coherent: H2's regression with a fifth covariate, the network-native coherence of each neuron's 50
    strongest examples (v6's top_pair_percentile). For the network reader this is partly circular,
    since its flag is built in the same embedding; Haiku and MiniLM never see that embedding."""
    defined = [r for r in rows if all(np.isfinite(r[c]) for c in WITH_COHERENCE)]
    covariates = np.column_stack([standardize(np.array([r[c] for r in defined], float)) for c in WITH_COHERENCE])
    models = sorted({r["model"] for r in defined})
    dummies = np.column_stack([[r["model"] == m for r in defined] for m in models[1:]]).astype(float)
    x, y = np.column_stack([covariates, dummies]), np.array([r[reader] for r in defined], float)

    def fit(xs, ys):
        return LogisticRegression(C=np.inf, max_iter=2000).fit(xs, ys).coef_[0][:len(WITH_COHERENCE)]

    point, draws = fit(x, y), []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() < y[idx].max():
            draws.append(fit(x[idx], y[idx]))
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {c: {"coefficient": float(p), "interval": [float(lo), float(hi)]}
            for c, p, lo, hi in zip(WITH_COHERENCE, point, low, high)}


def main() -> None:
    corpus = load_corpus()
    v5 = json.loads((RESULTS_DIR / "classification_v5.json").read_text())["neurons"]
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"]
    results = {"held_out": {}, "discovery": {}, "exploratory_with_coherence": {}}
    for d, depth in enumerate(DEPTHS):
        rows = [r for model in MODELS for r in detector_rows(model, depth, v5, v6, corpus)]
        target = "held_out" if depth in HELD_OUT_DEPTHS else "discovery"
        for k, reader in enumerate(("network", "minilm")):
            results[target][f"{reader}, {depth}"] = reader_tests(rows, reader, 10 * d + 2 * k)
            results["exploratory_with_coherence"][f"{reader}, {depth}"] = with_coherence(rows, reader, rng_for(MODELS[0], 100 + 10 * d + k))
    answers = json.loads(ANSWERS_PATH.read_text())
    haiku = haiku_rows(answers, v6, corpus)
    results["held_out"][f"haiku, {HAIKU_DEPTH} (fresh neurons)"] = reader_tests(haiku, "haiku", 90)
    results["exploratory_with_coherence"][f"haiku, {HAIKU_DEPTH} (fresh neurons)"] = with_coherence(haiku, "haiku", rng_for(MODELS[0], 190))
    (RESULTS_DIR / "classification_v12.json").write_text(json.dumps(results, indent=1))
    for part in ("held_out", "discovery"):
        print(part)
        for key, r in results[part].items():
            print(f"  {key:36} n={r['n']:5} quartiles {[round(x, 2) for x in r['by_quartile']]} "
                  f"H1 {r['h1']['holds']} ({r['h1']['top_minus_rest']:+.3f} {[round(x, 3) for x in r['h1']['interval']]}, "
                  f"{r['h1']['models_negative']}/6 negative)  H2 {r['h2']['holds']} "
                  f"(log_peak {r['h2']['log_peak']['coefficient']:+.2f} {[round(x, 2) for x in r['h2']['log_peak']['interval']]}; "
                  f"purity {r['h2']['token_purity']['coefficient']:+.2f}, peakedness {r['h2']['peakedness']['coefficient']:+.2f}, "
                  f"density {r['h2']['density']['coefficient']:+.2f})")
    print("exploratory, after review: H2 with coherence added")
    for key, r in results["exploratory_with_coherence"].items():
        print(f"  {key:36} " + ", ".join(f"{c} {r[c]['coefficient']:+.2f} {[round(x, 2) for x in r[c]['interval']]}"
                                          for c in ("log_peak", "coherence")))


if __name__ == "__main__":
    main()
