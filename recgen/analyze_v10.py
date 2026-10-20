"""Apply the pre-registered v10 rules (recgen/rubric_v10.py): stronger language-model readers on v7's units.

Reads cached data only: v6's recordings, v5's MiniLM embeddings, Haiku's v7 answers
(results/v7_llm_answers.json) and the v10 readers' answers (results/v10_<model>_answers.json, from
recgen.llm_reader_v10). A unit a reader never labeled is left out of that reader's rates and of
its paired comparisons, and counted.
"""

import json
from collections import Counter

import numpy as np

from recgen import analyze_v7
from recgen.analyze_v6 import MiniLM
from recgen.analyze_v7 import GROUPS
from recgen.config import RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.facets import display_text
from recgen.llm_reader import ANSWERS_PATH as HAIKU_ANSWERS_PATH
from recgen.llm_reader_v10 import answers_path
from recgen.rubric_v5 import MODELS
from recgen.rubric_v6 import cohen_kappa
from recgen.rubric_v7 import ONE_JOB, UNRELATED_JOBS
from recgen.rubric_v10 import (
    BASELINE, READERS, excess_interval, kappa_ceiling, kappa_interval, multiple_jobs, paired_difference,
    paired_kappa_difference,
)

LLM_READERS = (BASELINE,) + READERS
ALL_READERS = ("minilm", "network") + LLM_READERS
EXAMPLES_SHOWN = 2  # per model, of each kind of disagreement worth reading


def rng(k: int):
    return np.random.default_rng([SEED, 10, k])


def load_answers() -> dict:
    answers = {BASELINE: json.loads(HAIKU_ANSWERS_PATH.read_text())}
    for reader in READERS:
        path = answers_path(reader)
        answers[reader] = json.loads(path.read_text()) if path.exists() else {"first": {}, "retest": {}}
    return answers


def unit_rows(minilm: MiniLM, answers: dict) -> dict:
    """Each model's units with every reader's flag and label (None where a reader gave no label),
    checked against v7's committed per-unit flags."""
    committed = json.loads((RESULTS_DIR / "classification_v7.json").read_text())["units"]
    rows = {}
    for model in MODELS:
        model_rows = analyze_v7.flags_for(model, minilm, answers[BASELINE])
        for r, old in zip(model_rows, committed[model], strict=True):
            assert (r["id"], r["minilm"], r["network"], r["llm"]) == (old["id"], old["minilm"], old["network"], old["llm"])
            r[BASELINE], r[f"{BASELINE}_label"] = r.pop("llm"), r.pop("llm_label")
            for reader in READERS:
                label = answers[reader]["first"].get(r["id"], {}).get("label")
                r[f"{reader}_label"], r[reader] = label, None if label is None else label == UNRELATED_JOBS
        rows[model] = model_rows
    return rows


def as_llm(rows: list[dict], reader: str) -> list[dict]:
    """Rows with one language-model reader in v7's "llm" slot, so v7's statistics apply unchanged."""
    return [{**r, "llm": r[reader], "llm_label": r[f"{reader}_label"]} for r in rows]


def flags_of(rows: list[dict], reader: str, g: str, flag=None) -> list:
    """A reader's flags on one group; flag maps a label to a flag (default: v7's C flag)."""
    if flag is None:
        return [r[reader] for r in rows if r["group"] == g]
    return [flag(r[f"{reader}_label"]) for r in rows if r["group"] == g]


def v7_tests(rows: dict, reader: str) -> dict | None:
    """v7's four pre-registered tests for one reader, or None if a model has a group it never labeled."""
    if any(all(v is None for v in flags_of(rows[m], reader, g)) for m in MODELS for g in GROUPS):
        return None
    stats = [analyze_v7.model_stats(m, as_llm(rows[m], reader)) for m in MODELS]
    return {"verdicts": analyze_v7.verdicts(stats)["llm"],
            "per_model": {s["model"]: {"rates": s["rates"]["llm"], "power": s["power"]["llm"],
                                       "specificity": s["specificity"]["llm"],
                                       "kappa_with_network": s["kappa_real"]["network vs llm"],
                                       "labels": s["llm_labels"], "missing": s["llm_missing"]} for s in stats}}


def reader_excess(rows: dict, reader: str, seed: int, flag=None) -> dict:
    """The pre-registered over-all-models 50/50 excess for one reader, on the units it labeled."""
    pooled, real = [], []
    for m in MODELS:
        pooled.append([v for v in flags_of(rows[m], reader, "pooled 50/50", flag) if v is not None])
        real.append([v for v in flags_of(rows[m], reader, "real", flag) if v is not None])
    return {**excess_interval(pooled, real, rng(seed)), "units": sum(map(len, pooled)) + sum(map(len, real))}


def main_comparison(rows: dict, reader: str, seed: int, flag=None) -> dict:
    """Reader minus Haiku in the over-all-models 50/50 excess, paired on units both labeled."""
    pooled, real = [], []
    for m in MODELS:
        for g, out in (("pooled 50/50", pooled), ("real", real)):
            pairs = [(h, x) for h, x in zip(flags_of(rows[m], BASELINE, g, flag), flags_of(rows[m], reader, g, flag))
                     if h is not None and x is not None]
            out.append(np.array(pairs, float).reshape(-1, 2))
    if any(len(a) == 0 for a in pooled + real):
        return {"verdict": "not run", "units": sum(map(len, pooled)) + sum(map(len, real))}
    return {**paired_difference(pooled, real, rng(seed)), "units": sum(map(len, pooled)) + sum(map(len, real))}


def retest_pairs(answers: dict) -> dict:
    """(first, second) labels per retested unit, where both readings gave a label."""
    return {k: (answers["first"][k]["label"], v["label"]) for k, v in answers["retest"].items()
            if k in answers["first"] and answers["first"][k]["label"] is not None and v["label"] is not None}


def retest(answers: dict, seed: int) -> dict:
    pairs = retest_pairs(answers)
    if not pairs:
        return {"units": 0}
    first, second = (np.array([p[i] == UNRELATED_JOBS for p in pairs.values()]) for i in (0, 1))
    return {"units": len(pairs), "real_units": sum(":real:" in k for k in pairs),
            "same_label": float(np.mean([a == b for a, b in pairs.values()])),
            "same_flag": float(np.mean(first == second)),
            "flag_kappa": kappa_interval(first, second, rng(seed))}


def retest_versus_haiku(answers: dict, reader: str, seed: int) -> dict:
    haiku, other = retest_pairs(answers[BASELINE]), retest_pairs(answers[reader])
    shared = sorted(set(haiku) & set(other))
    if not shared:
        return {"units": 0}
    flags = {name: tuple(np.array([pairs[k][i] == UNRELATED_JOBS for k in shared]) for i in (0, 1))
             for name, pairs in (("haiku", haiku), ("reader", other))}
    return {"units": len(shared), **paired_kappa_difference(flags["haiku"], flags["reader"], rng(seed))}


def agreement(rows: dict, a: str, b: str, seed: int) -> dict:
    """Kappa between two readers on real neurons, per model and over all models."""
    per_model, pooled_a, pooled_b = {}, [], []
    for m in MODELS:
        pairs = [(x, y) for x, y in zip(flags_of(rows[m], a, "real"), flags_of(rows[m], b, "real"))
                 if x is not None and y is not None]
        if pairs:
            x, y = np.array(pairs, bool).T
            per_model[m] = cohen_kappa(x, y)
            pooled_a += list(x)
            pooled_b += list(y)
    if not pooled_a:
        return {"per_model": {}}
    return {"per_model": per_model, "all_models": kappa_interval(np.array(pooled_a), np.array(pooled_b), rng(seed))}


def label_shares(rows: dict, reader: str) -> dict:
    out = {}
    for m in MODELS + ["all"]:
        labels = [r[f"{reader}_label"] for model in (MODELS if m == "all" else [m]) for r in rows[model]
                  if r["group"] == "real" and r[f"{reader}_label"] is not None]
        counts = Counter(labels)
        out[m] = {label: round(counts[label] / len(labels), 3) for label in "ABC"} if labels else {}
    return out


def usage(answers: dict) -> dict:
    done = list(answers["first"].values()) + list(answers["retest"].values())
    return {"readings": len(done), "without_label": sum(a["label"] is None for a in done),
            "models_used": sorted({m for a in done for m in a["models"]}),
            "input_tokens": sum(a["input_tokens"] for a in done),
            "output_tokens": sum(a.get("output_tokens", 0) for a in done),
            "thinking_tokens": sum(a.get("thinking_tokens", 0) for a in done),
            "cost_usd_by_cli": round(sum(a.get("cost_usd") or 0 for a in done), 2)}


def summarize(rows: dict, answers: dict) -> dict:
    retests = {reader: retest(answers[reader], 10 + i) for i, reader in enumerate(LLM_READERS)}
    summary = {"main": {}, "secondary_multiple_jobs": {}, "excess": {}, "excess_multiple_jobs": {}, "v7_tests": {},
               "retest": retests, "retest_versus_haiku": {}, "agreement": {}, "label_shares": {}, "usage": {}}
    for i, reader in enumerate(ALL_READERS):
        summary["excess"][reader] = reader_excess(rows, reader, 20 + i)
    for i, reader in enumerate(LLM_READERS):
        summary["excess_multiple_jobs"][reader] = reader_excess(rows, reader, 30 + i, multiple_jobs)
        summary["v7_tests"][reader] = v7_tests(rows, reader)
        summary["label_shares"][reader] = label_shares(rows, reader)
        summary["usage"][reader] = usage(answers[reader])
    for i, reader in enumerate(READERS):
        summary["main"][reader] = main_comparison(rows, reader, 40 + i)
        summary["secondary_multiple_jobs"][reader] = main_comparison(rows, reader, 50 + i, multiple_jobs)
        summary["retest_versus_haiku"][reader] = retest_versus_haiku(answers, reader, 60 + i)
    retest_kappa = {reader: retests[reader]["flag_kappa"]["kappa"] for reader in LLM_READERS if retests[reader]["units"]}
    retest_kappa["network"] = 1.0  # deterministic
    pairs = [(x, "network") for x in LLM_READERS] + [(a, b) for i, a in enumerate(LLM_READERS) for b in LLM_READERS[i + 1:]]
    for i, (a, b) in enumerate(pairs):
        out = agreement(rows, a, b, 70 + i)
        if "all_models" in out and a in retest_kappa and b in retest_kappa:
            out["ceiling"] = kappa_ceiling(retest_kappa[a], retest_kappa[b])
            out["kappa_over_ceiling"] = out["all_models"]["kappa"] / out["ceiling"] if out["ceiling"] else None
        summary["agreement"][f"{a} vs {b}"] = out
    return summary


def write_report(rows: dict, answers: dict, corpus: np.ndarray) -> None:
    lines = ["# v10 results per model\n",
             "Flag rates (unrelated mixed) by reader and unit type at the middle layer, the language-model readers' "
             "labels on real neurons, then, per model, up to "
             f"{EXAMPLES_SHOWN} 50/50 pools where Opus and Haiku disagree in each direction and {EXAMPLES_SHOWN} real "
             "neurons where they disagree, with each reader's stated patterns and four of the unit's 50 snippets "
             "(its 1st, 2nd, 26th and 27th strongest; the readers saw all 50 in random order).\n"]
    for model in MODELS:
        model_rows = rows[model]
        lines.append(f"## pythia-{model}\n")
        lines.append("| Reader | " + " | ".join(GROUPS) + " |\n|---|" + "---|" * len(GROUPS))
        for reader in ALL_READERS:
            cells = []
            for g in GROUPS:
                values = [v for v in flags_of(model_rows, reader, g) if v is not None]
                cells.append(f"{np.mean(values):.0%}" if values else "n/a")
            lines.append(f"| {reader} | " + " | ".join(cells) + " |")
        for reader in LLM_READERS:
            counts = Counter(r[f"{reader}_label"] for r in model_rows if r["group"] == "real")
            lines.append(f"\n{reader} labels on real neurons: " + ", ".join(f"{k} {counts[k]}" for k in "ABC")
                         + (f", none {counts[None]}" if counts[None] else ""))
        lines.append("")
        for title, chosen in (
            ("50/50 pools Opus flags and Haiku calls one job",
             [r for r in model_rows if r["group"] == "pooled 50/50" and r["opus"] and r["haiku_label"] == ONE_JOB]),
            ("50/50 pools Haiku flags and Opus calls one job",
             [r for r in model_rows if r["group"] == "pooled 50/50" and r["haiku"] and r["opus_label"] == ONE_JOB]),
            ("Real neurons where Opus and Haiku disagree",
             [r for r in model_rows if r["group"] == "real" and r["opus"] is not None and r["opus"] != r["haiku"]])):
            lines.append(f"### {title}\n")
            for r in chosen[:EXAMPLES_SHOWN]:
                lines.append(f"**{r['id']}**, MLP neuron(s) {', '.join(map(str, r['neuron_ids']))} "
                             f"(network {r['network']}; labels: " +
                             ", ".join(f"{x} {r[f'{x}_label']}" for x in LLM_READERS) + ")\n")
                for reader in LLM_READERS:
                    reply = answers[reader]["first"].get(r["id"], {}).get("reply", "").replace("\n", " ")
                    lines.append(f"- {reader}: `{reply[:500]}`")
                lines += [f"- `{display_text(corpus, int(w), int(p))}`" for w, p in r["keys"][[0, 1, 25, 26]]]
                lines.append("")
    (RESULTS_DIR / "v10_models.md").write_text("\n".join(lines))


def main() -> None:
    corpus = load_corpus()
    answers = load_answers()
    rows = unit_rows(MiniLM(corpus), answers)
    summary = summarize(rows, answers)
    keep = ("id", "group", "neuron_ids", "shared_window", "minilm", "network") + \
        tuple(x for reader in LLM_READERS for x in (reader, f"{reader}_label"))
    units_out = {m: [{k: r[k] for k in keep} for r in model_rows] for m, model_rows in rows.items()}
    (RESULTS_DIR / "classification_v10.json").write_text(json.dumps({"summary": summary, "units": units_out}, indent=1))
    write_report(rows, answers, corpus)
    print(json.dumps({k: summary[k] for k in ("main", "secondary_multiple_jobs", "excess", "retest",
                                              "retest_versus_haiku")}, indent=1))


if __name__ == "__main__":
    main()
