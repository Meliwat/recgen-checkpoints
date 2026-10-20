"""Apply the pre-registered v22 rules (recgen/rubric_v22.py): v19's three readers, units and tests on v19's GPT-2
neurons recorded on OpenWebText, and the paired comparison with v19's results on the Pile.

Reads cached data only: data/v22/ (recgen.gpt2_v22), results/v22_llm_answers.json (recgen.read_v22),
results/classification_v19.json (v19's per-unit and per-neuron flags on the Pile), results/classification_v12.json
(through v19's summary) and results/v7_llm_answers.json (Haiku on Pythia's real neurons). MiniLM embeddings of every
example context and of the random positions are computed once and cached in data/v22/minilm.npz.
"""

import json
from collections import Counter

import numpy as np

from recgen import position_v19
from recgen.analyze_v6 import Layer
from recgen.analyze_v7 import READERS
from recgen.analyze_v12 import with_coherence
from recgen.analyze_v19 import (
    excess_by_reader, layer_checks, load_minilm, model_stats, neuron_rows, reader_tests, retest_interval, rng,
    unit_rows, without_draws,
)
from recgen.analyze_v7 import retest_stats
from recgen.config import CONTEXT_LEN, RESULTS_DIR
from recgen.corpus import packed_stream
from recgen.gpt2_v19 import tokenizer
from recgen.gpt2_v22 import V22_DIR, documents, load_owt_corpus, maxima_path, network_path, sampled_top_examples
from recgen.read_v19 import unit_prompt
from recgen.read_v22 import ANSWERS_PATH
from recgen.review_v19 import example_stats, prompt_stats, prompts_reproduce, real_rates
from recgen.rubric_v5 import DEAD_FRACTION, DENSE_MAX_WINDOWS, DENSE_STRONG_FRACTION, eligible
from recgen.rubric_v19 import DEPTH, MODELS, decide_a, rng_for
from recgen.rubric_v22 import PYTHIA_SIZE_MATCHED, SECONDARY_POSITION_VARIANT, decide_c, excess_change, rate_change
from recgen.rubric_v22 import rng as v22_rng

MINILM_PATH = V22_DIR / "minilm.npz"
V19_PATH = RESULTS_DIR / "classification_v19.json"
DETECTORS = ("network", "minilm")


def corpus_stats(corpus: np.ndarray) -> dict:
    """The sample's size as a packed stream, and the windows drawn from it."""
    stream = packed_stream(tokenizer(), documents())
    boundary = corpus == tokenizer().eos_token_id
    return {"documents": int((stream == tokenizer().eos_token_id).sum()), "stream_tokens": len(stream),
            "full_windows": len(stream) // CONTEXT_LEN, "windows": len(corpus), "window_tokens": int(corpus.size),
            "windows_with_a_document_boundary": int(boundary.any(axis=1).sum())}


def sample_eligibility(model: str, neurons: list[dict]) -> dict:
    """v5's eligibility rule on OpenWebText, applied to v19's sampled neurons (which v22 keeps whatever it says)."""
    window_max = np.load(maxima_path(model))["max"].astype(np.float32)
    peak = window_max.max(axis=0)
    ids = np.array([n["neuron"] for n in neurons])
    near_dead = peak[ids] < DEAD_FRACTION * np.median(peak)
    dense = (window_max[:, ids] > DENSE_STRONG_FRACTION * peak[ids]).mean(axis=0) > DENSE_MAX_WINDOWS
    return {"sampled_near_dead": int(near_dead.sum()), "sampled_dense": int(dense.sum()),
            "sampled_ineligible": int((~np.isin(ids, eligible(window_max))).sum()),
            "sampled_peak_not_positive": int((peak[ids] <= 0).sum()),
            "sampled_peakedness_undefined": int(sum(not np.isfinite(n["peakedness"]) for n in neurons))}


def paired_flags(owt: list[dict], pile: list[dict], key: str, reader: str) -> tuple[np.ndarray, np.ndarray]:
    """One reader's flags on the same units (or neurons) in both corpora, matched by key, where neither is missing."""
    pile_flags = {r[key]: r[reader] for r in pile}
    assert {r[key] for r in owt} == set(pile_flags), "v22's units are v19's"
    pairs = [(r[reader], pile_flags[r[key]]) for r in owt if r[reader] is not None and pile_flags[r[key]] is not None]
    return tuple(np.array(side, float) for side in zip(*pairs))


def in_group(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r["group"] == name]


def comparison(units: dict, neurons: dict, v19: dict, v7_answers: dict) -> dict:
    """Tests (c) and (d), and the paired secondary rates, OpenWebText against v19's Pile results."""
    def unit_pairs(group: str, reader: str) -> list[tuple[np.ndarray, np.ndarray]]:
        return [paired_flags(in_group(units[m], group), in_group(v19["units"][m], group), "id", reader) for m in MODELS]

    real = {reader: unit_pairs("real", reader) for reader in READERS}
    pooled = {reader: unit_pairs("pooled 50/50", reader) for reader in READERS}
    change = rate_change(real["llm"], v22_rng(0))
    owt_rate, pile_rate = (float(np.mean([pair[i].mean() for pair in real["llm"]])) for i in (0, 1))
    target = float(np.mean(list(real_rates(v7_answers["first"], PYTHIA_SIZE_MATCHED).values())))
    test_c = {"openwebtext": owt_rate, "pile": pile_rate, "pythia_size_matched": target,
              "change": change["change"], "interval": change["interval"],
              "by_model": {m: {"openwebtext": float(o.mean()), "pile": float(p.mean()), "neurons": len(o),
                               "interval": rate_change([(o, p)], v22_rng(60 + i))["interval"],
                               "flagged_only_on_openwebtext": int(((o == 1) & (p == 0)).sum()),
                               "flagged_only_on_pile": int(((o == 0) & (p == 1)).sum())}
                           for i, (m, (o, p)) in enumerate(zip(MODELS, real["llm"]))},
              **decide_c(owt_rate, pile_rate, target, change["interval"])}
    test_d = {reader: excess_change(list(zip(pooled[reader], real[reader])), v22_rng(10 + i))
              for i, reader in enumerate(READERS)}

    def summarized(pairs: list[tuple[np.ndarray, np.ndarray]], purpose: int) -> dict:
        out = rate_change(pairs, v22_rng(purpose))
        return {"openwebtext": float(np.mean([o.mean() for o, _ in pairs])), "pile": float(np.mean([p.mean() for _, p in pairs])),
                "change": out["change"], "interval": out["interval"],
                "by_model": {m: [float(o.mean()), float(p.mean())] for m, (o, p) in zip(MODELS, pairs)}}

    all_sampled = {d: [paired_flags(neurons[m], v19["neurons"][m], "neuron", d) for m in MODELS] for d in DETECTORS}
    paired_rates = {"real": {r: summarized(real[r], 20 + i) for i, r in enumerate(READERS)},
                    "pooled_50_50": {r: summarized(pooled[r], 30 + i) for i, r in enumerate(READERS)},
                    "all_sampled": {d: summarized(all_sampled[d], 40 + i) for i, d in enumerate(DETECTORS)}}
    labels = {m: {"openwebtext": dict(Counter(u["llm_label"] for u in in_group(units[m], "real"))),
                  "pile": dict(Counter(u["llm_label"] for u in in_group(v19["units"][m], "real")))} for m in MODELS}
    coherence = {m: {"openwebtext": float(np.median([n["coherence"] for n in neurons[m]])),
                     "pile": float(np.median([n["coherence"] for n in v19["neurons"][m]]))} for m in MODELS}
    return {"c": test_c, "d": test_d,
            "secondary": {"paired_rates": paired_rates, "haiku_labels_real": labels, "coherence_median": coherence}}


def position_secondary(position: dict, neurons: dict, units: dict, v19: dict) -> dict:
    """The network reader with token position projected out (position_v19's rule), beside v19's post hoc numbers."""
    variants = position_v19.summarize(position, neurons, units, lambda: rng(20), lambda: rng(1))
    pile = v19["summary"]["post_hoc_after_review"]["position"]["variants"]
    return {"secondary_variant": SECONDARY_POSITION_VARIANT, "openwebtext": variants,
            "pile_v19": {name: {k: v[k] for k in ("neuron_flags_changed", "h1", "network_excess")} for name, v in pile.items()},
            "null_position_correlation": {m: position[m]["null_position_correlation"] for m in MODELS},
            "position_r2": {m: position[m]["position_r2"] for m in MODELS}}


def main() -> None:
    corpus = load_owt_corpus()
    answers = json.loads(ANSWERS_PATH.read_text())
    v19 = json.loads(V19_PATH.read_text())
    v7_answers = json.loads((RESULTS_DIR / "v7_llm_answers.json").read_text())
    examples = {m: sampled_top_examples(m) for m in MODELS}
    minilm = load_minilm(corpus, [examples[m][1] for m in MODELS], MINILM_PATH)
    neurons, units, stats, checks, position, seen = {}, {}, [], {}, {}, {}
    for model in MODELS:
        layer = Layer(model, DEPTH, network_path(model))
        neurons[model] = neuron_rows(model, layer, minilm, corpus, examples[model], maxima_path(model))
        units[model] = unit_rows(model, layer, minilm, answers, neurons[model], network_path(model))
        stats.append(model_stats(model, units[model]))
        checks[model] = {**layer_checks(model, layer, neurons[model], maxima_path(model), network_path(model)),
                         **sample_eligibility(model, neurons[model])}
        position[model] = position_v19.model_sensitivity(layer, network_path(model), units[model], rng_for(model, 60))
        seen[model] = {**example_stats(units[model], corpus, tokenizer().decode),
                       **prompt_stats(units[model], corpus, unit_prompt, answers)}
        print(f"{model}: network flags {checks[model]['flag_rate_all_sampled']['network']:.3f} and MiniLM "
              f"{checks[model]['flag_rate_all_sampled']['minilm']:.3f} of {len(neurons[model])} sampled neurons; "
              f"{checks[model]['sampled_ineligible']} of them ineligible on OpenWebText", flush=True)

    owt = excess_by_reader(units, 0)
    test_a = {"openwebtext": {r: without_draws(owt[r]) for r in READERS}, "pile_v19": v19["summary"]["a"]["gpt2"],
              **decide_a({r: owt[r]["interval"] for r in READERS})}

    with_peak = [r for m in MODELS for r in neurons[m] if r["peak"] > 0]
    haiku = []
    for model in MODELS:
        for u in in_group(units[model], "real"):
            row = neurons[model][u["neurons"][0]]
            if row["peak"] > 0:
                haiku.append({**row, "llm": None if u["llm"] is None else float(u["llm"])})
    test_b = {"network": reader_tests(with_peak, "network", 20), "llm": reader_tests(haiku, "llm", 30),
              "left_out_peak_not_positive": sum(len(neurons[m]) for m in MODELS) - len(with_peak)}
    test_b["verdict"] = "holds" if test_b["network"]["h1"]["holds"] else "does not hold"
    test_b["pile_v19"] = {r: {"by_quartile": v19["summary"]["b"][r]["by_quartile"], "h1": v19["summary"]["b"][r]["h1"],
                              "log_peak": v19["summary"]["b"][r]["h2"]["log_peak"]} for r in ("network", "llm")}

    compared = comparison(units, neurons, v19, v7_answers)
    llm_rows = [a for part in ("first", "retest") for a in answers[part].values()]
    retest = {k: v for k, v in answers["retest"].items() if k in answers["first"]}
    summary = {
        "a": test_a, "b": test_b, "c": compared["c"], "d": compared["d"],
        "secondary": {
            **compared["secondary"],
            "position": position_secondary(position, neurons, units, v19),
            "minilm_magnitude": reader_tests(with_peak, "minilm", 40),
            "with_coherence": {"network": with_coherence(with_peak, "network", rng(50)),
                               "minilm": with_coherence(with_peak, "minilm", rng(51)),
                               "llm": with_coherence([r for r in haiku if r["llm"] is not None], "llm", rng(52))},
            "retest": {**retest_stats({"first": answers["first"], "retest": retest}),
                       "flag_kappa_interval": retest_interval(answers),
                       "pile_v19": v19["summary"]["secondary"]["retest"]},
            "models_with_power": {r: sum(s["power"][r]["has_power"] for s in stats) for r in READERS},
            "models_specific": {r: sum(s["specificity"][r]["specific"] for s in stats) for r in READERS},
            "examples_seen": {"openwebtext": seen,
                              "pile_v19": v19["summary"]["post_hoc_after_review"]["haiku"]["examples_seen"]["gpt2"]},
        },
        "checks": {
            "corpus": corpus_stats(corpus),
            "layers": checks,
            "largest_activation_error": max(c["activation_error"] for c in checks.values()),
            "largest_float32_error": max(c["float32_error"] for c in checks.values()),
            "minilm_null_percentiles_on_openwebtext": minilm.calibration(),
            "llm_models_used": sorted({m for a in llm_rows for m in a["models"]}),
            "llm_readings": len(llm_rows), "llm_without_label": sum(a["label"] is None for a in llm_rows),
            "llm_not_read": len(answers["not_read"]), "llm_input_tokens": sum(a["input_tokens"] for a in llm_rows),
            "llm_cost_usd": answers["cost_usd"],
            "prompts_reproduce": prompts_reproduce(units, corpus, answers),
        },
    }
    units_out = {model: [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items() if k not in ("rows", "keys")}
                         for r in rows] for model, rows in units.items()}
    neurons_out = {model: [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "model"}
                           for r in rows] for model, rows in neurons.items()}
    (RESULTS_DIR / "classification_v22.json").write_text(
        json.dumps({"summary": summary, "models": stats, "units": units_out, "neurons": neurons_out}, indent=1))

    brief = {"a": test_a, "b": {r: test_b[r]["h1"] | {"h2": test_b[r]["h2"]["log_peak"], "by_quartile": test_b[r]["by_quartile"]}
                                for r in ("network", "llm")} | {"verdict": test_b["verdict"]},
             "c": {k: v for k, v in compared["c"].items() if k != "by_model"}, "d": compared["d"],
             "checks": {k: v for k, v in summary["checks"].items() if k != "layers"}}
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
