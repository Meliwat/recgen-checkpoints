"""Apply the pre-registered v19 rules (recgen/rubric_v19.py): v7's three readers on v7's units, and v12's
magnitude tests, at the middle layer of GPT-2 small, medium and large, beside Pythia's results.

Reads cached data only: data/v19/ (recgen.gpt2_v19), results/v19_llm_answers.json (recgen.read_v19), and
Pythia's results/classification_v7.json (per-unit flags) and classification_v12.json. MiniLM embeddings of
every example context and of the random positions are computed once and cached in data/v19/minilm.npz.

Checks added after an independent review, none pre-registered, go under "post_hoc_after_review": the network
reader with token position projected out (recgen.position_v19, which also reads Pythia's data/v6/*_middle.npz),
and checks on Haiku's readings and the recording (recgen.review_v19, which also reads v11's cache in data/v11/
if it's there).
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recgen import position_v19, review_v19
from recgen.analyze_v6 import PRIMARY, Layer, MiniLM, detect
from recgen.analyze_v7 import GROUPS, READERS, group, max_kappa, rate, retest_stats
from recgen.analyze_v12 import with_coherence
from recgen.config import CONTEXT_LEN, RESULTS_DIR, SEED
from recgen.facets import embed
from recgen.gpt2_v19 import V19_DIR, context_text, load_gpt2_corpus, maxima_path, network_path, sampled_top_examples
from recgen.read_v19 import ANSWERS_PATH
from recgen.rubric_v5 import DEAD_FRACTION, DENSE_MAX_WINDOWS, DENSE_STRONG_FRACTION, eligible
from recgen.rubric_v6 import MERGE_PERCENTILE, cohen_kappa, pair_similarities, percentile_rank, random_positions
from recgen.rubric_v7 import UNRELATED_JOBS, has_power, is_specific
from recgen.rubric_v12 import COVARIATES, h2, neuron_stats, quartiles
from recgen.rubric_v19 import DEPTH, MODELS, decide_a, difference, excess_over_models, h1, rng_for
from recgen.units_v19 import build_units

MINILM_PATH = V19_DIR / "minilm.npz"
POST_HOC_NOTE = "post hoc, added after an independent review; not pre-registered"
EMBED_CHUNK = 50_000
THREADS = 8
RETEST_BOOTSTRAP = 2000  # as v7's retest interval
DETECTORS = ("network", "minilm")


def rng(purpose: int):
    """Generators for statistics pooled over models."""
    return np.random.default_rng([SEED, 19, 100, purpose])


class GPT2MiniLM(MiniLM):
    """v5's detector with analyze_v6.MiniLM's rules, on MiniLM embeddings of GPT-2's contexts."""

    def __init__(self, keys: np.ndarray, vectors: np.ndarray, random_vectors: np.ndarray):
        self.codes = keys[:, 0].astype(np.int64) * CONTEXT_LEN + keys[:, 1]
        assert np.all(np.diff(self.codes) > 0)
        self.vectors = vectors
        self.null = pair_similarities(random_vectors.astype(np.float64))


def load_minilm(corpus: np.ndarray, example_keys: list[np.ndarray]) -> GPT2MiniLM:
    """MiniLM embeddings of every example context (float16, as v5's cache) and of the random positions."""
    keys = np.unique(np.concatenate([k.reshape(-1, 2) for k in example_keys]), axis=0)
    if not (MINILM_PATH.exists() and np.array_equal(np.load(MINILM_PATH)["keys"], keys)):
        vectors = np.concatenate([embed([context_text(corpus, int(w), int(p)) for w, p in keys[i:i + EMBED_CHUNK]])
                                  for i in range(0, len(keys), EMBED_CHUNK)])
        random = embed([context_text(corpus, int(w), int(p)) for w, p in random_positions(len(corpus))])
        np.savez(MINILM_PATH, keys=keys, vectors=vectors.astype(np.float16), random=random)
    cached = np.load(MINILM_PATH)
    return GPT2MiniLM(keys, cached["vectors"], cached["random"])


def covariates(model: str, neurons: np.ndarray, corpus: np.ndarray) -> dict[int, dict]:
    """v12's covariates for each sampled neuron, with log peak relative to the layer's median sampled peak."""
    data = np.load(maxima_path(model))
    window_max, window_pos = data["max"], data["pos"]
    stats = {}
    for n in map(int, neurons):
        column = window_max[:, n].astype(np.float32)
        top = np.argsort(-column, kind="stable")[:50]
        stats[n] = neuron_stats(column, [int(corpus[w, window_pos[w, n]]) for w in top])
    median_peak = np.median([s["peak"] for s in stats.values()])
    for s in stats.values():
        s["log_peak"] = float(np.log(s["peak"] / median_peak))
    return stats


def network_neuron(layer: Layer, s: int) -> dict:
    """v6's detector on sampled neuron s, and the coherence of its top 50 (v6's top-pair percentile)."""
    units, null, levels = layer.embedded(PRIMARY, layer.rows[s], [s])
    result = detect(units, levels["merge"][MERGE_PERCENTILE], levels["unrelated"])
    return {"network": float(result["unrelated_mixed"]), "network_facets": result["n_facets"],
            "coherence": percentile_rank(float(np.median(pair_similarities(units))), null)}


def neuron_rows(model: str, layer: Layer, minilm: GPT2MiniLM, corpus: np.ndarray) -> list[dict]:
    """Every sampled neuron with its covariates, peak quartile and both detectors' flags."""
    neurons, keys, _ = sampled_top_examples(model)
    assert np.array_equal(neurons, layer.neurons) and np.array_equal(keys, layer.keys[layer.rows])
    with ThreadPoolExecutor(THREADS) as pool:
        network = list(pool.map(lambda s: network_neuron(layer, s), range(len(neurons))))
    stats = covariates(model, neurons, corpus)
    quartile = quartiles(np.array([stats[int(n)]["peak"] for n in neurons]))
    return [{"model": model, "neuron": int(n), "quartile": int(quartile[s]), **stats[int(n)], **network[s],
             "minilm": float(minilm.detect(keys[s])["unrelated_mixed"])} for s, n in enumerate(neurons)]


def unit_rows(model: str, layer: Layer, minilm: GPT2MiniLM, answers: dict, neurons: list[dict]) -> list[dict]:
    """Each unit of one model with every reader's flag (None where Haiku gave no label), as analyze_v7."""
    rows = []
    for unit in build_units(model):
        units, _, levels = layer.embedded(PRIMARY, unit["rows"], list(unit["neurons"]))
        network = bool(detect(units, levels["merge"][MERGE_PERCENTILE], levels["unrelated"])["unrelated_mixed"])
        if unit["kind"] == "real":  # the same computation as the neuron's own flag
            assert network == bool(neurons[unit["neurons"][0]]["network"])
        label = answers["first"].get(unit["id"], {}).get("label")
        rows.append({**unit, "group": group(unit), "neuron_ids": [int(layer.neurons[n]) for n in unit["neurons"]],
                     "shared_window": len({int(w) for w, _ in unit["keys"]}) < len(unit["keys"]),
                     "minilm": bool(minilm.detect(unit["keys"])["unrelated_mixed"]), "network": network,
                     "llm": None if label is None else label == UNRELATED_JOBS, "llm_label": label})
    return rows


def model_stats(model: str, rows: list[dict]) -> dict:
    """analyze_v7.model_stats with v19's seeds: rates by unit type, v7's power and specificity tests, kappas."""
    by_group = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    stats = {"model": model, "rates": {}, "power": {}, "specificity": {}, "kappa_real": {}, "kappa_ceiling": {},
             "llm_labels": {g: dict(Counter(r["llm_label"] for r in members)) for g, members in by_group.items()},
             "llm_missing": sum(r["llm"] is None for r in rows),
             "shared_window_share": {g: float(np.mean([r["shared_window"] for r in members])) for g, members in by_group.items()}}
    for i, reader in enumerate(READERS):
        stats["rates"][reader] = {g: rate([r[reader] for r in members], rng_for(model, 20 + i)) for g, members in by_group.items()}
        real, pooled, related = ([r[reader] for r in by_group[g] if r[reader] is not None] for g in ("real", "pooled 50/50", "related"))
        powered, excess = has_power(np.array(pooled), np.array(real), rng_for(model, 30 + i))
        specific, gap = is_specific(np.array(related), np.array(pooled))
        stats["power"][reader] = {"has_power": powered, "excess": excess}
        stats["specificity"][reader] = {"specific": specific, "gap": gap}
    real_rows = [r for r in by_group["real"] if r["llm"] is not None]
    for a, b in (("minilm", "network"), ("minilm", "llm"), ("network", "llm")):
        x, y = np.array([r[a] for r in real_rows]), np.array([r[b] for r in real_rows])
        stats["kappa_real"][f"{a} vs {b}"] = cohen_kappa(x, y)
        stats["kappa_ceiling"][f"{a} vs {b}"] = max_kappa(float(x.mean()), float(y.mean()))
    return stats


def excess_by_reader(units_by_model: dict[str, list[dict]], purpose: int) -> dict:
    """Test (a)'s measure for each reader: per model, the 50/50 pools' and real units' flags."""
    out = {}
    for i, reader in enumerate(READERS):
        per_model = [tuple(np.array([r[reader] for r in rows if r["group"] == g and r[reader] is not None], float)
                           for g in ("pooled 50/50", "real")) for rows in units_by_model.values()]
        out[reader] = excess_over_models(per_model, rng(purpose + i))
    return out


def reader_tests(rows: list[dict], reader: str, purpose: int) -> dict:
    """v12's H1 and H2 for one reader, with v19's three-model H1."""
    rows = [r for r in rows if r[reader] is not None]
    groups = []
    for model in MODELS:
        r = [x for x in rows if x["model"] == model]
        groups.append((np.array([x["log_peak"] for x in r]), np.array([x["quartile"] for x in r]),
                       np.array([x[reader] for x in r], float)))
    regression_rows = [{**{k: x[k] for k in ("model",) + COVARIATES}, "flag": float(x[reader])} for x in rows]
    return {"n": len(rows),
            "by_quartile": [float(np.mean([x[reader] for x in rows if x["quartile"] == q])) for q in range(4)],
            "by_quartile_by_model": {m: [float(f[q == k].mean()) for k in range(4)] for m, (_, q, f) in zip(MODELS, groups)},
            "h1": h1(groups, rng(purpose)), "h2": h2(regression_rows, rng(purpose + 1))}


def retest_interval(answers: dict) -> list[float]:
    """The retest flag kappa's 95% bootstrap interval, as v7's."""
    pairs = [(answers["first"][k]["label"], v["label"]) for k, v in answers["retest"].items()
             if k in answers["first"] and answers["first"][k]["label"] is not None and v["label"] is not None]
    first, second = (np.array([p[i] == UNRELATED_JOBS for p in pairs]) for i in (0, 1))
    generator, kappas = rng(90), []
    for _ in range(RETEST_BOOTSTRAP):
        idx = generator.integers(0, len(pairs), len(pairs))
        kappas.append(cohen_kappa(first[idx], second[idx]))
    return [round(float(x), 3) for x in np.nanpercentile(kappas, [2.5, 97.5])]


def layer_checks(model: str, layer: Layer, neurons: list[dict]) -> dict:
    window_max = np.load(maxima_path(model))["max"].astype(np.float32)
    peak = window_max.max(axis=0)
    data = np.load(network_path(model))
    return {"layer": layer.layer, "n_layers": layer.n_layers, "width": window_max.shape[1],
            "eligible": len(eligible(window_max)), "near_dead": int((peak < DEAD_FRACTION * np.median(peak)).sum()),
            "dense": int(((window_max > DENSE_STRONG_FRACTION * peak).mean(axis=0) > DENSE_MAX_WINDOWS).sum()),
            "sampled": len(neurons), "activation_error": layer.activation_error,
            "float32_error": float(data["float32_error"]), "median_peak": float(np.median([n["peak"] for n in neurons])),
            "mean_share": layer.mean_share,
            "flag_rate_all_sampled": {d: float(np.mean([n[d] for n in neurons])) for d in DETECTORS}}


def without_draws(result: dict) -> dict:
    return {"excess": result["excess"], "interval": result["interval"]}


def main() -> None:
    corpus = load_gpt2_corpus()
    answers = json.loads(ANSWERS_PATH.read_text())
    minilm = load_minilm(corpus, [sampled_top_examples(m)[1] for m in MODELS])
    neurons, units, stats, checks, position, recording = {}, {}, [], {}, {}, {}
    for model in MODELS:
        layer = Layer(model, DEPTH, network_path(model))
        neurons[model] = neuron_rows(model, layer, minilm, corpus)
        units[model] = unit_rows(model, layer, minilm, answers, neurons[model])
        stats.append(model_stats(model, units[model]))
        checks[model] = layer_checks(model, layer, neurons[model])
        position[model] = position_v19.model_sensitivity(layer, network_path(model), units[model], rng_for(model, 60))
        recording[model] = {"largest_coordinate": review_v19.largest_coordinate(layer.nulls["uncentered"].vectors),
                            "token_purity_ties": review_v19.purity_ties(model, neurons[model], corpus)}
        print(f"{model}: layer {layer.layer}, network flags {checks[model]['flag_rate_all_sampled']['network']:.3f} and "
              f"MiniLM {checks[model]['flag_rate_all_sampled']['minilm']:.3f} of {len(neurons[model])} sampled neurons", flush=True)

    v7 = json.loads((RESULTS_DIR / "classification_v7.json").read_text())
    gpt2, pythia = excess_by_reader(units, 0), excess_by_reader(v7["units"], 10)
    test_a = {"gpt2": {r: without_draws(gpt2[r]) for r in READERS}, "pythia_v7": {r: without_draws(pythia[r]) for r in READERS},
              **decide_a({r: gpt2[r]["interval"] for r in READERS}),
              "secondary_gpt2_minus_pythia": {r: difference(gpt2[r], pythia[r]) for r in READERS}}

    all_neurons = [r for model in MODELS for r in neurons[model]]
    haiku = []
    for model in MODELS:
        for u in units[model]:
            if u["group"] == "real":
                haiku.append({**neurons[model][u["neurons"][0]], "llm": None if u["llm"] is None else float(u["llm"])})
    test_b = {"network": reader_tests(all_neurons, "network", 20), "llm": reader_tests(haiku, "llm", 30)}
    test_b["verdict"] = "replicates" if test_b["network"]["h1"]["holds"] else "does not replicate"
    v12 = json.loads((RESULTS_DIR / "classification_v12.json").read_text())
    pythia_b = {key: {"by_quartile": r["by_quartile"], "top_minus_rest": r["h1"]["top_minus_rest"], "interval": r["h1"]["interval"],
                      "log_peak_coefficient": r["h2"]["log_peak"]["coefficient"]}
                for part in ("discovery", "held_out") for key, r in v12[part].items()}

    llm_rows = [a for part in ("first", "retest") for a in answers[part].values()]
    summary = {
        "a": test_a,
        "b": {**test_b, "pythia_v12": pythia_b},
        "secondary": {
            "minilm_magnitude": reader_tests(all_neurons, "minilm", 40),
            "with_coherence": {"network": with_coherence(all_neurons, "network", rng(50)),
                               "minilm": with_coherence(all_neurons, "minilm", rng(51)),
                               "llm": with_coherence([r for r in haiku if r["llm"] is not None], "llm", rng(52))},
            "retest": {**retest_stats({"first": answers["first"], "retest": {k: v for k, v in answers["retest"].items() if k in answers["first"]}}),
                       "flag_kappa_interval": retest_interval(answers)},
            "models_with_power": {r: sum(s["power"][r]["has_power"] for s in stats) for r in READERS},
            "models_specific": {r: sum(s["specificity"][r]["specific"] for s in stats) for r in READERS},
        },
        "checks": {
            "layers": checks,
            "largest_activation_error": max(c["activation_error"] for c in checks.values()),
            "largest_float32_error": max(c["float32_error"] for c in checks.values()),
            "minilm_null_percentiles_on_gpt2": minilm.calibration(),
            "llm_models_used": sorted({m for a in llm_rows for m in a["models"]}),
            "llm_readings": len(llm_rows), "llm_without_label": sum(a["label"] is None for a in llm_rows),
            "llm_not_read": len(answers["not_read"]), "llm_input_tokens": sum(a["input_tokens"] for a in llm_rows),
            "llm_cost_usd": answers["cost_usd"],
        },
    }
    summary["post_hoc_after_review"] = {
        "note": POST_HOC_NOTE,
        "position": {
            "null_position_correlation": {"gpt2": {m: position[m]["null_position_correlation"] for m in MODELS},
                                          "pythia_v6_middle": position_v19.pythia_null_correlations()},
            "position_r2": {m: position[m]["position_r2"] for m in MODELS},
            # the network reader's own generators in tests (b) and (a), so only the flags differ
            "variants": position_v19.summarize(position, neurons, units, lambda: rng(20), lambda: rng(1)),
        },
        "haiku": review_v19.haiku_checks(units, corpus, answers),
        "recording": recording,
    }
    units_out = {model: [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items() if k not in ("rows", "keys")}
                         for r in rows] for model, rows in units.items()}
    neurons_out = {model: [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "model"}
                           for r in rows] for model, rows in neurons.items()}
    (RESULTS_DIR / "classification_v19.json").write_text(
        json.dumps({"summary": summary, "models": stats, "units": units_out, "neurons": neurons_out}, indent=1))

    print(json.dumps({"a": test_a, "b": {k: v["h1"] | {"h2": v["h2"]["log_peak"], "by_quartile": v["by_quartile"]}
                                         if isinstance(v, dict) else v for k, v in test_b.items()},
                      "checks": {k: v for k, v in summary["checks"].items() if k != "layers"},
                      "post_hoc_position": {name: {k: v[k] for k in ("neuron_flags_changed", "h1", "network_excess")}
                                            for name, v in summary["post_hoc_after_review"]["position"]["variants"].items()}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
