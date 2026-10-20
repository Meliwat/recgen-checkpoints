"""Apply the pre-registered v11 rules (recgen/rubric_v11.py): v7's three readers at the quarter and
three-quarters depths, and each reader's power compared across depths, with v7's middle depth.

Reads cached data only: v6's recordings (data/v6/), v5's MiniLM embeddings, the language model's
answers (results/v11_llm_answers.json, from recgen.llm_reader_v11), v6's per-neuron flags
(results/classification_v6.json) for the checks, and v7's per-unit flags
(results/classification_v7.json) for the middle depth. A unit the language model never labeled, or
never read, is left out of that reader's rates and counted.
"""

import json
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from recgen.analyze_v6 import PRIMARY, Layer, MiniLM, detect
from recgen.analyze_v7 import EXAMPLES_SHOWN, GROUPS, READERS, exploratory_stats, group, model_stats, retest_stats, verdicts
from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import display_text
from recgen.llm_reader_v11 import ANSWERS_PATH
from recgen.rubric_v5 import MODELS, decide
from recgen.rubric_v6 import MERGE_PERCENTILE
from recgen.rubric_v7 import UNRELATED_JOBS
from recgen.rubric_v11 import (
    ALL_DEPTHS, DEPTH_PAIRS, DEPTHS, N_READERS, V7_RESOLVED_MODEL, compare_depths, excess_against_size,
    excess_over_models, reader_dependence, rng_for,
)
from recgen.units_v11 import build_units

assert len(READERS) == N_READERS


def flags_for(model: str, depth: str, minilm: MiniLM, answers: dict) -> list[dict]:
    """Each unit of one model and depth with every reader's flag (None where the language model gave
    no label or hasn't read the unit)."""
    layer = Layer(model, depth)
    rows = []
    for unit in build_units(model, depth):
        units, _, levels = layer.embedded(PRIMARY, unit["rows"], list(unit["neurons"]))
        network = detect(units, levels["merge"][MERGE_PERCENTILE], levels["unrelated"])
        answer = answers["first"].get(unit["id"])
        label = None if answer is None else answer["label"]
        rows.append({**unit, "group": group(unit), "neuron_ids": [int(layer.neurons[n]) for n in unit["neurons"]],
                     "shared_window": len({int(w) for w, _ in unit["keys"]}) < len(unit["keys"]),
                     "minilm": bool(minilm.detect(unit["keys"])["unrelated_mixed"]),
                     "network": bool(network["unrelated_mixed"]),
                     "llm": None if label is None else label == UNRELATED_JOBS, "llm_label": label,
                     "llm_read": answer is not None})
    return rows


def excess_by_reader(rows: dict, depth: str) -> dict:
    """The pre-registered over-all-models excess for each reader at one depth. rows: model -> units."""
    out = {}
    for i, reader in enumerate(READERS):
        per_model = [(np.array([r[reader] for r in model_rows if r["group"] == "pooled 50/50" and r[reader] is not None]),
                      np.array([r[reader] for r in model_rows if r["group"] == "real" and r[reader] is not None]))
                     for model_rows in rows.values()]
        out[reader] = excess_over_models(per_model, rng_for(depth, i))
    return out


def depth_summary(depth: str, stats: list[dict], excess: dict, scale: dict) -> dict:
    """Per-depth results beside the per-model tests: the excess, its trend with size, and whether the
    scale trend depends on the reader. scale: reader -> verdict dict from analyze_v7.verdicts."""
    per_model = {reader: [s["power"][reader]["excess"] for s in stats] for reader in READERS}
    return {
        "verdicts": scale,
        "excess_over_models": {r: {"excess": round(e["excess"], 4), "interval": [round(x, 4) for x in e["interval"]]}
                               for r, e in excess.items()},
        "excess_per_model": {r: [round(x, 4) for x in v] for r, v in per_model.items()},
        "excess_against_size_spearman": {r: round(excess_against_size(v), 3) for r, v in per_model.items()},
        "reader_dependence": reader_dependence({r: scale[r]["scale"] for r in READERS},
                                               {r: excess[r]["interval"] for r in READERS}),
        "kappa_real": {pair: [round(s["kappa_real"][pair], 3) for s in stats] for pair in stats[0]["kappa_real"]},
        "real_rates": {r: [round(s["rates"][r]["real"]["share"], 3) for s in stats] for r in READERS},
    }


def middle_from_v7() -> tuple[dict, list[dict], dict]:
    """v7's middle depth: its per-unit flags, per-model stats and verdicts, read from its results."""
    v7 = json.loads((RESULTS_DIR / "classification_v7.json").read_text())
    return v7["units"], v7["models"], v7["summary"]["verdicts"]


def with_llm_labels(rows: dict) -> dict:
    """Only the units the language model labeled, for v7's checks, which assume every unit has a label.
    rows: model -> units."""
    return {model: [r for r in model_rows if r["llm"] is not None] for model, model_rows in rows.items()}


def rounded(value):
    """A float, or a list of floats, to 4 decimals for the results file; anything else unchanged."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [round(x, 4) for x in value]
    return value


def checks(rows: dict, answers: dict, v6: dict) -> dict:
    """Real-neuron flags against v5's and v6's, the resolved model, and readings missing a label.
    v6: results/classification_v6.json."""
    layers = {(s["model"], s["depth"]): s for s in v6["layers"]}
    out = {"real_flags_differ_from_v5_v6": {}, "subsample_vs_full_sample": {}, "llm_missing": {}}
    for depth in DEPTHS:
        differ = {"minilm": 0, "network": 0}
        full = {"minilm": [], "network": [], "v11 minilm": [], "v11 network": []}
        for model, model_rows in rows[depth].items():
            per_neuron = v6["neurons"][f"{model} {depth}"]
            real = [r for r in model_rows if r["group"] == "real"]
            for r in real:
                neuron = per_neuron[str(r["neuron_ids"][0])]
                differ["minilm"] += r["minilm"] != neuron["v5_unrelated_mixed"]
                differ["network"] += r["network"] != neuron["unrelated_mixed"]
            full["minilm"].append(round(layers[model, depth]["v5_unrelated_mixed_share"], 3))
            full["network"].append(round(layers[model, depth]["unrelated_mixed"]["share"], 3))
            full["v11 minilm"].append(round(float(np.mean([r["minilm"] for r in real])), 3))
            full["v11 network"].append(round(float(np.mean([r["network"] for r in real])), 3))
        out["real_flags_differ_from_v5_v6"][depth] = differ
        out["subsample_vs_full_sample"][depth] = full
        first = answers[depth]["first"]
        out["llm_missing"][depth] = {
            "not read": sum(not r["llm_read"] for model_rows in rows[depth].values() for r in model_rows),
            "no valid label": sum(a["label"] is None for a in first.values()),
            "retests read": len(answers[depth]["retest"])}
    models_used = sorted({m for part in answers.values() for reading in part.values() for a in reading.values()
                          for m in a["models"]})
    out["llm_models_used"] = models_used
    out["llm_model_matches_v7"] = models_used == [V7_RESOLVED_MODEL]
    out["llm_input_tokens"] = sum(a["input_tokens"] for part in answers.values() for reading in part.values()
                                  for a in reading.values())
    return out


def exploratory(stats: dict, by_depth: dict, full_sample: dict) -> dict:
    """Not pre-registered: checks added after seeing the v11 results, and (the full-sample reader
    dependence) after an independent review of them. stats: depth -> per-model stats, smallest model
    first (the middle depth from v7); by_depth: the per-depth summaries; full_sample: v6's decisions,
    which hold v5's MiniLM and v6's network verdicts on their full samples of about 2,048 neurons per model."""
    out = {"scale_without_70m": {}, "pooled_rate_against_size_spearman": {}, "llm_one_job_share_real": {},
           "reader_dependence_with_full_samples": {}}
    for depth, per_model in stats.items():
        out["scale_without_70m"][depth], out["pooled_rate_against_size_spearman"][depth] = {}, {}
        for reader in READERS:
            real = [s["rates"][reader]["real"] for s in per_model][1:]
            verdict, rho = decide([r["share"] for r in real], [tuple(r["interval"]) for r in real])
            out["scale_without_70m"][depth][reader] = {"verdict": verdict, "spearman": round(rho, 3)}
            pooled = [s["rates"][reader]["pooled 50/50"]["share"] for s in per_model]
            out["pooled_rate_against_size_spearman"][depth][reader] = round(
                float(spearmanr(np.arange(len(pooled)), pooled).statistic), 3)
        # Labels of real neurons; JSON turns a missing label's None key into "null".
        labeled = [{k: v for k, v in s["llm_labels"]["real"].items() if k not in (None, "null")} for s in per_model]
        out["llm_one_job_share_real"][depth] = [round(c.get("A", 0) / sum(c.values()), 3) for c in labeled]
        decision = full_sample[f"{depth}: unrelated_mixed"]
        out["reader_dependence_with_full_samples"][depth] = reader_dependence(
            {"minilm": decision["v5"]["verdict"], "network": decision["v6"]["verdict"],
             "llm": by_depth[depth]["verdicts"]["llm"]["scale"]},
            {r: by_depth[depth]["excess_over_models"][r]["interval"] for r in READERS})
    return out


def write_report(stats: dict, rows: dict, answers: dict, corpus: np.ndarray) -> None:
    lines = ["# v11 results per model\n",
             "Flag rates (unrelated mixed) by reader and unit type at the quarter and three-quarters depths, then, "
             f"per model, up to {EXAMPLES_SHOWN} real neurons each reader alone flags and {EXAMPLES_SHOWN} 50/50 "
             "pools the language model calls one job, with its stated patterns and four of the unit's 50 snippets "
             "(its 1st, 2nd, 26th and 27th strongest; the language model saw all 50 in random order). Unit ids use "
             "v6's sample indices; the MLP neuron ids are given beside them. The middle depth is in "
             "[v7_models.md](v7_models.md).\n"]
    for depth in DEPTHS:
        lines.append(f"## {depth}\n")
        for s in stats[depth]:
            model = s["model"]
            lines.append(f"### pythia-{model}, {depth}\n")
            lines.append("| Reader | " + " | ".join(GROUPS) + " |\n|---|" + "---|" * len(GROUPS))
            for reader in READERS:
                lines.append(f"| {reader} | " + " | ".join(f"{s['rates'][reader][g]['share']:.0%}" for g in GROUPS) + " |")
            lines.append(f"\nLanguage-model labels: {s['llm_labels']}; without a label: {s['llm_missing']}\n")
            model_rows = rows[depth][model]
            for title, chosen in (
                ("Real neurons only the language model flags",
                 [r for r in model_rows if r["group"] == "real" and r["llm"] and not r["minilm"] and not r["network"]]),
                ("Real neurons only the network reader flags",
                 [r for r in model_rows if r["group"] == "real" and r["network"] and r["llm"] is False and not r["minilm"]]),
                ("50/50 pools the language model calls one job",
                 [r for r in model_rows if r["group"] == "pooled 50/50" and r["llm_label"] == "A"])):
                lines.append(f"#### {title}\n")
                for r in chosen[:EXAMPLES_SHOWN]:
                    reply = answers[depth]["first"][r["id"]]["reply"].replace("\n", " ")
                    lines.append(f"**{r['id']}**, MLP neuron(s) {', '.join(map(str, r['neuron_ids']))} "
                                 f"(MiniLM {r['minilm']}, network {r['network']}, LLM {r['llm_label']})\n")
                    lines.append(f"- Language model: `{reply[:700]}`")
                    lines += [f"- `{display_text(corpus, int(w), int(p))}`" for w, p in r["keys"][[0, 1, 25, 26]]]
                    lines.append("")
    (RESULTS_DIR / "v11_models.md").write_text("\n".join(lines))


def main() -> None:
    corpus = load_corpus()
    answers = json.loads(ANSWERS_PATH.read_text())
    minilm = MiniLM(corpus)
    rows = {depth: {model: flags_for(model, depth, minilm, answers[depth]) for model in MODELS} for depth in DEPTHS}
    stats = {depth: [model_stats(model, rows[depth][model]) for model in MODELS] for depth in DEPTHS}
    v7_units, v7_stats, v7_verdicts = middle_from_v7()
    excess = {depth: excess_by_reader(rows[depth], depth) for depth in DEPTHS}
    excess["middle"] = excess_by_reader(v7_units, "middle")
    by_depth = {depth: depth_summary(depth, stats[depth], excess[depth], verdicts(stats[depth])) for depth in DEPTHS}
    by_depth["middle"] = depth_summary("middle", v7_stats, excess["middle"], v7_verdicts)
    by_depth = {depth: by_depth[depth] for depth in ALL_DEPTHS}
    for depth in DEPTHS:
        by_depth[depth]["retest"] = retest_stats(answers[depth])
        explored = exploratory_stats(with_llm_labels(rows[depth]), answers[depth])
        by_depth[depth]["retest"]["flag_kappa_interval"] = explored["retest"]["flag_kappa_interval"]
        by_depth[depth]["llm_labels_real"] = dict(Counter(r["llm_label"] for model_rows in rows[depth].values()
                                                          for r in model_rows if r["group"] == "real"))
        # v7's post-review checks, repeated (its own excess is replaced by the pre-registered one above).
        by_depth[depth]["v7_review_checks"] = {k: v for k, v in explored.items() if k not in ("pooled_excess", "retest")}
    comparison = {reader: {f"{a} vs {b}": {k: rounded(v) for k, v in
                                           compare_depths(excess[a][reader], excess[b][reader]).items()}
                           for a, b in DEPTH_PAIRS} for reader in READERS}
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())
    all_stats = {d: v7_stats if d == "middle" else stats[d] for d in ALL_DEPTHS}
    summary = {"by_depth": by_depth, "depth_comparison": comparison, "checks": checks(rows, answers, v6),
               "exploratory": exploratory(all_stats, by_depth, v6["summary"]["decisions"])}
    units_out = {depth: {model: [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()
                                  if k not in ("rows", "keys")} for r in model_rows]
                         for model, model_rows in rows[depth].items()} for depth in DEPTHS}
    (RESULTS_DIR / "classification_v11.json").write_text(json.dumps(
        {"summary": summary, "models": stats, "units": units_out}, indent=1))
    write_report(stats, rows, answers, corpus)
    print(json.dumps({"depth_comparison": comparison, "checks": summary["checks"], "exploratory": summary["exploratory"],
                      **{d: {k: by_depth[d][k] for k in ("verdicts", "excess_over_models", "reader_dependence")}
                         for d in ALL_DEPTHS}}, indent=1))


if __name__ == "__main__":
    main()
