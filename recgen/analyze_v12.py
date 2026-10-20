"""Apply the pre-registered v12 rules (recgen/rubric_v12.py): does a neuron's peak activation predict how
mixed it looks, on data the v9 finding didn't use?

Reads cached data only: v5's window maxima and flags, v6's flags, and v12's fresh Haiku answers.
The middle-layer network and MiniLM results are where the effect was found, so they're reported as
"discovery", apart from the held-out tests.
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.read_v12 import ANSWERS_PATH, fresh_units
from recgen.rubric_v5 import DEPTHS, MODELS
from recgen.rubric_v7 import N_REAL, UNRELATED_JOBS
from recgen.rubric_v12 import (BOOTSTRAP, COVARIATES, HAIKU_DEPTH, HELD_OUT_DEPTHS, h1, h2, neuron_stats, quartiles,
                                rng_for, standardize)
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path
from recgen.units_v9 import build_unions

UNRELATED_SIMILARITY = 0.1  # v5's cutoff for MiniLM's flag
WITH_COHERENCE = COVARIATES + ("coherence",)
V11_DIR = DATA_DIR / "v11" / "llm"  # v11's cached Haiku readings (branch v11), read with v7's prompt and CLI


def layer_covariates(model: str, depth: str, neuron_ids: list[int], corpus: np.ndarray) -> dict[int, dict]:
    data = np.load(sweep_path(model, depth))
    window_max, window_pos = data["max"], data["pos"]
    stats = {}
    for n in neuron_ids:
        column = window_max[:, n].astype(np.float32)
        top = np.argsort(-column, kind="stable")[:50]
        stats[n] = neuron_stats(column, [int(corpus[w, window_pos[w, n]]) for w in top])
        stats[n]["top50_mean"] = float(column[top].mean())
    median_peak = np.median([s["peak"] for s in stats.values()])
    median_top50 = np.median([s["top50_mean"] for s in stats.values()])
    for s in stats.values():
        s["log_peak"] = float(np.log(s["peak"] / median_peak))
        s["log_top50_mean"] = float(np.log(s["top50_mean"] / median_top50))
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


def v11_rows(depth: str, detector: list[dict]) -> list[dict] | None:
    """Pre-registered: v11's Haiku readings of real neurons at a held-out depth, with the same covariates
    and peak quartiles as the detector rows. None until v11 has read every model's real neurons there."""
    by_neuron = {(r["model"], r["neuron"]): r for r in detector}
    rows = []
    for model in MODELS:
        files = sorted(V11_DIR.glob(f"{model}_{depth.replace(' ', '_')}_real__*.json"))
        if len(files) < N_REAL:
            return None
        neurons = np.load(v6_path(model, depth))["neurons"]
        for f in files:
            answer = json.loads(f.read_text())
            if answer["label"] is not None:
                s = int(answer["id"].rsplit(":", 1)[1])
                rows.append({**by_neuron[model, int(neurons[s])], "haiku": float(answer["label"] == UNRELATED_JOBS)})
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
    models = sorted({r["model"] for r in defined})
    dummies = np.column_stack([[r["model"] == m for r in defined] for m in models[1:]]).astype(float)
    y = np.array([r[reader] for r in defined], float)

    def design_with(covariates):
        return np.column_stack([standardize(np.array([r[c] for r in defined], float)) for c in covariates] + [dummies])

    def fit(xs, ys, n_covariates):
        return LogisticRegression(C=np.inf, max_iter=2000).fit(xs, ys).coef_[0][:n_covariates]

    x, x_h2 = design_with(WITH_COHERENCE), design_with(COVARIATES)
    point = fit(x, y, len(WITH_COHERENCE))
    change = point[0] - fit(x_h2, y, len(COVARIATES))[0]
    draws, changes = [], []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() < y[idx].max():
            draws.append(fit(x[idx], y[idx], len(WITH_COHERENCE)))
            changes.append(draws[-1][0] - fit(x_h2[idx], y[idx], len(COVARIATES))[0])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    out = {c: {"coefficient": float(p), "interval": [float(lo), float(hi)]}
           for c, p, lo, hi in zip(WITH_COHERENCE, point, low, high)}
    # added after v12's review: the paired change in log peak's coefficient from H2's model, and the collinearity
    out["log_peak_change_from_h2"] = {"change": float(change), "interval": np.percentile(changes, [2.5, 97.5]).tolist()}
    out["log_peak_coherence_correlation"] = float(np.corrcoef([r["log_peak"] for r in defined],
                                                              [r["coherence"] for r in defined])[0, 1])
    return out


def by_top50_mean(rows: list[dict], reader: str, thresholds: dict[str, np.ndarray], rng) -> dict:
    """Exploratory, added after v12's review: H1 with quartiles of the mean of a neuron's 50 strongest window
    maxima (the level of the examples the readers see) instead of its single largest one."""
    groups, quartile_flags = [], []
    for model in MODELS:
        r = [x for x in rows if x["model"] == model]
        values = np.array([x["log_top50_mean"] for x in r])
        q = np.searchsorted(thresholds[model], values, side="right")
        flags = np.array([x[reader] for x in r])
        groups.append((values, q, flags))
        quartile_flags += list(zip(q, flags))
    by_quartile = [float(np.mean([f for q, f in quartile_flags if q == k])) for k in range(4)]
    return {"n": len(rows), "by_quartile": by_quartile, "h1": h1(groups, rng)}


def top50_thresholds(detector: list[dict]) -> dict[str, np.ndarray]:
    return {model: np.quantile([r["log_top50_mean"] for r in detector if r["model"] == model], [0.25, 0.5, 0.75])
            for model in MODELS}


def rank_50_fraction(detector: list[dict]) -> list[list[float]]:
    """Exploratory, added after v12's review: by peak quartile, the range over models of the median
    fraction of the peak at which a neuron's 50th-strongest window sits (how far down its own range the
    top 50 reaches)."""
    by_model = [[float(np.nanmedian([1 / r["peakedness"] for r in detector if r["model"] == model and r["quartile"] == q]))
                 for model in MODELS] for q in range(4)]
    return [[round(min(q), 3), round(max(q), 3)] for q in by_model]


def union_neurons() -> set[tuple[str, int]]:
    """The neurons inside v9's Haiku-read unions at the middle depth."""
    return {(model, int(n)) for model in MODELS for u in build_unions(model)[0] for n in u["neuron_ids"]}


def main() -> None:
    corpus = load_corpus()
    v5 = json.loads((RESULTS_DIR / "classification_v5.json").read_text())["neurons"]
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"]
    results = {"held_out": {}, "discovery": {}, "exploratory_with_coherence": {},
               "exploratory_after_v12_review": {"top50_mean_quartiles": {}, "rank_50_fraction_of_peak": {}}}
    thresholds = {}
    for d, depth in enumerate(DEPTHS):
        rows = [r for model in MODELS for r in detector_rows(model, depth, v5, v6, corpus)]
        thresholds[depth] = top50_thresholds(rows)
        results["exploratory_after_v12_review"]["rank_50_fraction_of_peak"][depth] = rank_50_fraction(rows)
        target = "held_out" if depth in HELD_OUT_DEPTHS else "discovery"
        for k, reader in enumerate(("network", "minilm")):
            results[target][f"{reader}, {depth}"] = reader_tests(rows, reader, 10 * d + 2 * k)
            results["exploratory_with_coherence"][f"{reader}, {depth}"] = with_coherence(rows, reader, rng_for(MODELS[0], 100 + 10 * d + k))
            results["exploratory_after_v12_review"]["top50_mean_quartiles"][f"{reader}, {depth}"] = by_top50_mean(
                rows, reader, thresholds[depth], rng_for(MODELS[0], 200 + 10 * d + k))
        v11 = v11_rows(depth, rows) if depth in HELD_OUT_DEPTHS else None
        if v11 is not None:
            key = f"haiku, {depth} (v11 real neurons)"
            results["held_out"][key] = reader_tests(v11, "haiku", 60 + 2 * d)
            results["exploratory_with_coherence"][key] = with_coherence(v11, "haiku", rng_for(MODELS[0], 160 + d))
            results["exploratory_after_v12_review"]["top50_mean_quartiles"][key] = by_top50_mean(
                v11, "haiku", thresholds[depth], rng_for(MODELS[0], 260 + d))
    answers = json.loads(ANSWERS_PATH.read_text())
    haiku = haiku_rows(answers, v6, corpus)
    fresh = f"haiku, {HAIKU_DEPTH} (fresh neurons)"
    results["held_out"][fresh] = reader_tests(haiku, "haiku", 90)
    results["exploratory_with_coherence"][fresh] = with_coherence(haiku, "haiku", rng_for(MODELS[0], 190))
    results["exploratory_after_v12_review"]["top50_mean_quartiles"][fresh] = by_top50_mean(
        haiku, "haiku", thresholds[HAIKU_DEPTH], rng_for(MODELS[0], 290))
    in_unions = union_neurons()
    outside = [r for r in haiku if (r["model"], r["neuron"]) not in in_unions]
    results["exploratory_after_v12_review"]["fresh_neurons_outside_v9_unions"] = {
        "left_out": len(haiku) - len(outside), **reader_tests(outside, "haiku", 92)}
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
                                          for c in ("log_peak", "coherence"))
              + f", log peak change {r['log_peak_change_from_h2']['change']:+.2f} "
                f"{[round(x, 2) for x in r['log_peak_change_from_h2']['interval']]}, r {r['log_peak_coherence_correlation']:.2f}")
    print("exploratory, after v12's review: H1 by quartile of the top-50 mean")
    for key, r in results["exploratory_after_v12_review"]["top50_mean_quartiles"].items():
        print(f"  {key:36} quartiles {[round(x, 2) for x in r['by_quartile']]} top minus rest {r['h1']['top_minus_rest']:+.3f} "
              f"{[round(x, 3) for x in r['h1']['interval']]}, {r['h1']['models_negative']}/6 negative")
    r = results["exploratory_after_v12_review"]["fresh_neurons_outside_v9_unions"]
    print(f"fresh neurons outside v9's unions: {r['left_out']} left out, n={r['n']}, H1 {r['h1']['top_minus_rest']:+.3f} "
          f"{[round(x, 3) for x in r['h1']['interval']]}, H2 log peak {r['h2']['log_peak']['coefficient']:+.2f} "
          f"{[round(x, 2) for x in r['h2']['log_peak']['interval']]}")


if __name__ == "__main__":
    main()
