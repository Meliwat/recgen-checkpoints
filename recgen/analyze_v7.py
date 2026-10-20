"""Apply the pre-registered v7 rules (recgen/rubric_v7.py): three readers on the same units.

Reads cached data only: v6's recordings (data/v6/), v5's MiniLM embeddings, and the language
model's answers (results/v7_llm_answers.json, from recgen.llm_reader). A unit the language model
never labeled is left out of that reader's rates and counted.
"""

import json
from collections import Counter
from math import comb

import numpy as np

from recgen.analyze_v6 import PRIMARY, Layer, MiniLM, detect
from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import display_text
from recgen.llm_reader import ANSWERS_PATH
from recgen.rubric_v5 import MODELS, bootstrap_interval, decide
from recgen.rubric_v6 import MERGE_PERCENTILE, cohen_kappa
from recgen.rubric_v7 import (
    DEPTH, MIN_MODELS, MIXES, N_POOLED, N_REAL, POWER_MARGIN, UNRELATED_JOBS, has_power, is_specific, rng_for,
)
from recgen.units_v7 import build_units

READERS = ("minilm", "network", "llm")
GROUPS = ["real"] + [f"pooled {mix}" for mix in MIXES] + ["related"]
EXAMPLES_SHOWN = 2  # per model, of each kind of disagreement worth reading
# Exploratory, added after an independent review of the first v7 results.
SIMULATED_EXCESSES = (0.15, 0.21, 0.25, 0.30)
SIMULATED_BASE_RATE = 0.45
SIMULATIONS = 500
BOOTSTRAP = 2000


def group(unit: dict) -> str:
    return "real" if unit["kind"] == "real" else "related" if unit["kind"] == "related" else f"pooled {unit['mix']}"


def flags_for(model: str, minilm: MiniLM, answers: dict) -> list[dict]:
    """Each unit of one model with every reader's flag (None where the language model gave no label)."""
    layer = Layer(model, DEPTH)
    rows = []
    for unit in build_units(model):
        units, _, levels = layer.embedded(PRIMARY, unit["rows"], list(unit["neurons"]))
        network = detect(units, levels["merge"][MERGE_PERCENTILE], levels["unrelated"])
        label = answers["first"][unit["id"]]["label"]
        rows.append({**unit, "group": group(unit), "neuron_ids": [int(layer.neurons[n]) for n in unit["neurons"]],
                     "shared_window": len({int(w) for w, _ in unit["keys"]}) < len(unit["keys"]),
                     "minilm": bool(minilm.detect(unit["keys"])["unrelated_mixed"]),
                     "network": bool(network["unrelated_mixed"]),
                     "llm": None if label is None else label == UNRELATED_JOBS, "llm_label": label})
    return rows


def rate(values: list, rng) -> dict:
    values = np.array([v for v in values if v is not None], bool)
    low, high = bootstrap_interval(values, rng)
    return {"share": float(values.mean()), "interval": [low, high], "n": len(values)}


def model_stats(model: str, rows: list[dict]) -> dict:
    by_group = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    stats = {"model": model, "rates": {}, "power": {}, "specificity": {}, "kappa_real": {},
             "llm_labels": {g: dict(Counter(r["llm_label"] for r in members)) for g, members in by_group.items()},
             "llm_missing": sum(r["llm"] is None for r in rows)}
    for i, reader in enumerate(READERS):
        stats["rates"][reader] = {g: rate([r[reader] for r in members], rng_for(model, 20 + i)) for g, members in by_group.items()}
        real = [r[reader] for r in by_group["real"] if r[reader] is not None]
        pooled = [r[reader] for r in by_group["pooled 50/50"] if r[reader] is not None]
        related = [r[reader] for r in by_group["related"] if r[reader] is not None]
        powered, excess = has_power(np.array(pooled), np.array(real), rng_for(model, 30 + i))
        specific, gap = is_specific(np.array(related), np.array(pooled))
        stats["power"][reader] = {"has_power": powered, "excess": excess}
        stats["specificity"][reader] = {"specific": specific, "gap": gap}
    real_rows = [r for r in by_group["real"] if r["llm"] is not None]
    for a, b in (("minilm", "network"), ("minilm", "llm"), ("network", "llm")):
        stats["kappa_real"][f"{a} vs {b}"] = cohen_kappa(np.array([r[a] for r in real_rows]), np.array([r[b] for r in real_rows]))
    return stats


def retest_stats(answers: dict) -> dict:
    pairs = [(answers["first"][k]["label"], v["label"]) for k, v in answers["retest"].items()]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    first, second = (np.array([p[i] == UNRELATED_JOBS for p in pairs]) for i in (0, 1))
    return {"units": len(pairs), "same_label": float(np.mean([a == b for a, b in pairs])),
            "same_flag": float(np.mean(first == second)), "flag_kappa": cohen_kappa(first, second)}


def verdicts(stats: list[dict]) -> dict:
    out = {}
    for reader in READERS:
        real = [s["rates"][reader]["real"] for s in stats]
        scale, rho = decide([r["share"] for r in real], [tuple(r["interval"]) for r in real])
        out[reader] = {
            "power": sum(s["power"][reader]["has_power"] for s in stats) >= MIN_MODELS,
            "models_with_power": sum(s["power"][reader]["has_power"] for s in stats),
            "specific": sum(s["specificity"][reader]["specific"] for s in stats) >= MIN_MODELS,
            "models_specific": sum(s["specificity"][reader]["specific"] for s in stats),
            "scale": scale, "scale_spearman": rho,
        }
    return out


def max_kappa(a: float, b: float) -> float:
    """The highest Cohen's kappa two flag sets with these rates can reach."""
    chance = a * b + (1 - a) * (1 - b)
    return (1 - abs(a - b) - chance) / (1 - chance) if chance < 1 else float("nan")


def pass_probability(excess: float, rng) -> float:
    """How often the pre-registered power test passes in one model for a reader whose true 50/50 rate
    is excess above a SIMULATED_BASE_RATE real rate, with v7's sample sizes."""
    passes = 0
    for _ in range(SIMULATIONS):
        real = rng.random(N_REAL) < SIMULATED_BASE_RATE
        pooled = rng.random(N_POOLED) < SIMULATED_BASE_RATE + excess
        passes += has_power(pooled, real, rng)[0]
    return passes / SIMULATIONS


def exploratory_stats(rows: dict, answers: dict) -> dict:
    """Not pre-registered: checks added after an independent review of the first v7 results."""
    real = [r for model_rows in rows.values() for r in model_rows if r["group"] == "real"]
    out = {"kappa_ceiling": {}, "flag_rate_given_minilm": {}, "shared_window_share": {}, "pooled_excess": {},
           "models_with_excess_at_margin": {}, "power_test_pass_probability": {}}
    for model, model_rows in rows.items():
        r = [x for x in model_rows if x["group"] == "real"]
        rate_of = {reader: float(np.mean([x[reader] for x in r])) for reader in READERS}
        out["kappa_ceiling"][model] = {f"minilm vs {b}": round(max_kappa(rate_of["minilm"], rate_of[b]), 3)
                                       for b in ("network", "llm")}
    for reader in ("network", "llm"):
        out["flag_rate_given_minilm"][reader] = {
            "minilm flagged": round(float(np.mean([x[reader] for x in real if x["minilm"]])), 3),
            "minilm not flagged": round(float(np.mean([x[reader] for x in real if not x["minilm"]])), 3)}
    for g in GROUPS:
        members = [x for model_rows in rows.values() for x in model_rows if x["group"] == g]
        out["shared_window_share"][g] = round(float(np.mean([x["shared_window"] for x in members])), 3)
    related = [x for model_rows in rows.values() for x in model_rows if x["group"] == "related"]
    out["llm_flag_rate_in_related_pools"] = {
        "with a shared window": round(float(np.mean([x["llm"] for x in related if x["shared_window"]])), 3),
        "without": round(float(np.mean([x["llm"] for x in related if not x["shared_window"]])), 3)}
    rng = np.random.default_rng([7, 99])
    for reader in READERS:
        per_model = []
        for model_rows in rows.values():
            per_model.append((np.array([x[reader] for x in model_rows if x["group"] == "pooled 50/50"], float),
                              np.array([x[reader] for x in model_rows if x["group"] == "real"], float)))
        point = float(np.mean([p.mean() - q.mean() for p, q in per_model]))
        draws = [np.mean([rng.choice(p, len(p)).mean() - rng.choice(q, len(q)).mean() for p, q in per_model])
                 for _ in range(BOOTSTRAP)]
        out["pooled_excess"][reader] = {"mean_over_models": round(point, 3),
                                        "interval": [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])]}
        out["models_with_excess_at_margin"][reader] = int(sum(p.mean() - q.mean() >= POWER_MARGIN - 1e-9 for p, q in per_model))
    for excess in SIMULATED_EXCESSES:
        p = pass_probability(excess, rng)
        five_of_six = sum(comb(6, k) * p ** k * (1 - p) ** (6 - k) for k in (5, 6))
        out["power_test_pass_probability"][f"+{excess:.0%} over {SIMULATED_BASE_RATE:.0%}"] = {
            "one model": round(p, 3), "5 or more of 6": round(five_of_six, 3)}
    pairs = [(answers["first"][k]["label"], v["label"], k) for k, v in answers["retest"].items()]
    first, second = (np.array([p[i] == UNRELATED_JOBS for p in pairs]) for i in (0, 1))
    kappas = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(pairs), len(pairs))
        kappas.append(cohen_kappa(first[idx], second[idx]))
    out["retest"] = {"flag_kappa_interval": [round(float(x), 3) for x in np.nanpercentile(kappas, [2.5, 97.5])],
                     "real_units": sum(":real:" in k for _, _, k in pairs)}
    return out


def write_report(stats: list[dict], rows: dict, answers: dict, corpus: np.ndarray) -> None:
    lines = ["# v7 results per model\n",
             "Flag rates (unrelated mixed) by reader and unit type at the middle layer, then, per model, "
             f"up to {EXAMPLES_SHOWN} real neurons each reader alone flags and {EXAMPLES_SHOWN} 50/50 pools the "
             "language model missed, with its stated patterns and four of the unit's 50 snippets (its 1st, 2nd, "
             "26th and 27th strongest; the language model saw all 50 in random order). Unit ids use v6's sample "
             "indices; the MLP neuron ids are given beside them.\n"]
    for s in stats:
        model = s["model"]
        lines.append(f"## pythia-{model}\n")
        lines.append("| Reader | " + " | ".join(GROUPS) + " |\n|---|" + "---|" * len(GROUPS))
        for reader in READERS:
            cells = [f"{s['rates'][reader][g]['share']:.0%}" for g in GROUPS]
            lines.append(f"| {reader} | " + " | ".join(cells) + " |")
        lines.append(f"\nLanguage-model labels: {s['llm_labels']}; without a label: {s['llm_missing']}\n")
        model_rows = rows[model]
        for title, chosen in (
            ("Real neurons only the language model flags",
             [r for r in model_rows if r["group"] == "real" and r["llm"] and not r["minilm"] and not r["network"]]),
            ("Real neurons only the network reader flags",
             [r for r in model_rows if r["group"] == "real" and r["network"] and not r["llm"] and not r["minilm"]]),
            ("50/50 pools the language model calls one job",
             [r for r in model_rows if r["group"] == "pooled 50/50" and r["llm_label"] == "A"])):
            lines.append(f"### {title}\n")
            for r in chosen[:EXAMPLES_SHOWN]:
                reply = answers["first"][r["id"]]["reply"].replace("\n", " ")
                lines.append(f"**{r['id']}**, MLP neuron(s) {', '.join(map(str, r['neuron_ids']))} "
                             f"(MiniLM {r['minilm']}, network {r['network']}, LLM {r['llm_label']})\n")
                lines.append(f"- Language model: `{reply[:700]}`")
                lines += [f"- `{display_text(corpus, int(w), int(p))}`" for w, p in r["keys"][[0, 1, 25, 26]]]
                lines.append("")
    (RESULTS_DIR / "v7_models.md").write_text("\n".join(lines))


def main() -> None:
    corpus = load_corpus()
    answers = json.loads(ANSWERS_PATH.read_text())
    minilm = MiniLM(corpus)
    rows = {model: flags_for(model, minilm, answers) for model in MODELS}
    stats = [model_stats(model, rows[model]) for model in MODELS]
    summary = {"verdicts": verdicts(stats), "retest": retest_stats(answers),
               "exploratory": exploratory_stats(rows, answers),
               "llm_models_used": sorted({m for a in answers["first"].values() for m in a["models"]}),
               "llm_input_tokens": sum(a["input_tokens"] for part in answers.values() for a in part.values())}
    units_out = {model: [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items() if k not in ("rows", "keys")}
                         for r in model_rows] for model, model_rows in rows.items()}
    (RESULTS_DIR / "classification_v7.json").write_text(json.dumps({"summary": summary, "models": stats,
                                                                    "units": units_out}, indent=1))
    write_report(stats, rows, answers, corpus)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
