"""Apply the pre-registered v7 rules (recgen/rubric_v7.py): three readers on the same units.

Reads cached data only: v6's recordings (data/v6/), v5's MiniLM embeddings, and the language
model's answers (results/v7_llm_answers.json, from recgen.llm_reader). A unit the language model
never labeled is left out of that reader's rates and counted.
"""

import json
from collections import Counter

import numpy as np

from recgen.analyze_v6 import PRIMARY, Layer, MiniLM, detect
from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import display_text
from recgen.llm_reader import ANSWERS_PATH
from recgen.rubric_v5 import MODELS, bootstrap_interval, decide
from recgen.rubric_v6 import MERGE_PERCENTILE, cohen_kappa
from recgen.rubric_v7 import (
    DEPTH, MIN_MODELS, MIXES, UNRELATED_JOBS, has_power, is_specific, rng_for,
)
from recgen.units_v7 import build_units

READERS = ("minilm", "network", "llm")
GROUPS = ["real"] + [f"pooled {mix}" for mix in MIXES] + ["related"]
EXAMPLES_SHOWN = 2  # per model, of each kind of disagreement worth reading


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
        rows.append({**unit, "group": group(unit),
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


def write_report(stats: list[dict], rows: dict, answers: dict, corpus: np.ndarray) -> None:
    lines = ["# v7 results per model\n",
             "Flag rates (unrelated mixed) by reader and unit type at the middle layer, then, per model, "
             f"up to {EXAMPLES_SHOWN} real neurons each reader alone flags and {EXAMPLES_SHOWN} 50/50 pools the "
             "language model missed, with its stated patterns.\n"]
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
                lines.append(f"**{r['id']}** (MiniLM {r['minilm']}, network {r['network']}, LLM {r['llm_label']})\n")
                lines.append(f"- Language model: `{reply[:400]}`")
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
