"""Apply the pre-registered v12 rules (recgen/rubric_v12.py): does a neuron's peak activation predict how
mixed it looks, on data the v9 finding didn't use?

Reads cached data only: v5's window maxima and flags, v6's flags, and v12's fresh Haiku answers.
The middle-layer network and MiniLM results are where the effect was found, so they're reported as
"discovery", apart from the held-out tests.
"""

import json

import numpy as np

from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.read_v12 import ANSWERS_PATH, fresh_units
from recgen.rubric_v5 import DEPTHS, MODELS
from recgen.rubric_v7 import UNRELATED_JOBS
from recgen.rubric_v12 import HAIKU_DEPTH, HELD_OUT_DEPTHS, h1, h2, neuron_stats, quartiles, rng_for
from recgen.sweep import sweep_path

UNRELATED_SIMILARITY = 0.1  # v5's cutoff for MiniLM's flag


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
             "network": float(network[str(n)]["unrelated_mixed"]),
             "minilm": float(minilm[str(n)]["n_facets"] >= 2 and minilm[str(n)]["least_related_similarity"] < UNRELATED_SIMILARITY)}
            for n in ids]


def haiku_rows(answers: dict, corpus: np.ndarray) -> list[dict]:
    rows = []
    for model in MODELS:
        units = fresh_units(model)
        covariates = layer_covariates(model, HAIKU_DEPTH, [u["neuron_id"] for u in units], corpus)
        # log peak relative to all sampled neurons of the layer, as for the detectors
        layer_peaks = np.load(sweep_path(model, HAIKU_DEPTH))["max"].astype(np.float32)
        sampled = [int(n) for n in json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"][f"{model} {HAIKU_DEPTH}"]]
        median_peak = np.median(layer_peaks[:, sampled].max(axis=0))
        for u in units:
            label = answers[u["id"]]["label"]
            if label is None:
                continue
            c = dict(covariates[u["neuron_id"]])
            c["log_peak"] = float(np.log(c["peak"] / median_peak))
            rows.append({"model": model, "neuron": u["neuron_id"], "quartile": u["quartile"], **c,
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


def main() -> None:
    corpus = load_corpus()
    v5 = json.loads((RESULTS_DIR / "classification_v5.json").read_text())["neurons"]
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"]
    results = {"held_out": {}, "discovery": {}}
    for d, depth in enumerate(DEPTHS):
        rows = [r for model in MODELS for r in detector_rows(model, depth, v5, v6, corpus)]
        target = "held_out" if depth in HELD_OUT_DEPTHS else "discovery"
        for k, reader in enumerate(("network", "minilm")):
            results[target][f"{reader}, {depth}"] = reader_tests(rows, reader, 10 * d + 2 * k)
    answers = json.loads(ANSWERS_PATH.read_text())
    results["held_out"][f"haiku, {HAIKU_DEPTH} (fresh neurons)"] = reader_tests(haiku_rows(answers, corpus), "haiku", 90)
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


if __name__ == "__main__":
    main()
