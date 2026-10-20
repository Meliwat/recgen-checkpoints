"""Apply the pre-registered v8 rules (recgen/rubric_v8.py): two investigating agents beside v7's
three readers, on the same units.

Reads results files only: the agents' readings (results/v8_agent_answers.json, from
recgen.agent_v8) and v7's per-unit flags and labels (results/classification_v7.json). A unit an
agent never labeled is left out of that agent's rates and counted.
"""

import json
from collections import Counter

import numpy as np

from recgen.agent_v8 import ANSWERS_PATH
from recgen.config import RESULTS_DIR
from recgen.rubric_v5 import bootstrap_interval
from recgen.rubric_v6 import cohen_kappa
from recgen.rubric_v7 import ONE_JOB, UNRELATED_JOBS
from recgen.rubric_v8 import (
    AGENT_MODEL, BOOTSTRAP, MIN_TOOL_CALLS, MODELS, SECOND_AGENT_MODEL, excess, gain_verdict, paired_gain, parse_answer,
    rng_for,
)

AGENTS = {"haiku agent": AGENT_MODEL, "sonnet agent": SECOND_AGENT_MODEL}
LLM_READERS = ("haiku", "haiku agent", "sonnet agent")  # "haiku" is v7's one-shot reader
READERS = ("minilm", "network") + LLM_READERS
GROUPS = ("real", "pooled 50/50", "related")
MEASURES = {"flag": "unrelated jobs (C)", "not_one_job": "not one job (B or C)"}
PRIMARY = ("haiku agent", "haiku")
GAINS = (PRIMARY, ("sonnet agent", "haiku"), ("sonnet agent", "haiku agent"))
ESSAY_ONE_JOB = 0.9
EXAMPLES_SHOWN = 2
# Post hoc, added after reading the first results.
LABEL_RANK = {"A": 0, "B": 1, "C": 2}
CATCH_ALL_WORDS = ("predictab", "formulaic", "fixed", "templat", "boilerplate", "continuation")


def unit_rows(answers: dict, v7_units: dict) -> dict[str, list[dict]]:
    """Each v8 unit of each model with every reader's label and flags (None where a reader gave no label)."""
    by_id = {u["id"]: u for model_units in v7_units.values() for u in model_units}
    rows = {model: [] for model in MODELS}
    for unit_id in answers[AGENT_MODEL]["first"]:
        unit = by_id[unit_id]
        labels = {"haiku": unit["llm_label"],
                  **{name: answers[arm]["first"].get(unit_id, {}).get("label") for name, arm in AGENTS.items()}}
        rows[unit["model"]].append({
            "id": unit_id, "model": unit["model"], "group": unit["group"], "neuron_ids": unit["neuron_ids"],
            "shared_window": unit["shared_window"], "labels": labels,
            "flag": {"minilm": unit["minilm"], "network": unit["network"],
                     **{r: None if label is None else label == UNRELATED_JOBS for r, label in labels.items()}},
            "not_one_job": {r: None if label is None else label != ONE_JOB for r, label in labels.items()}})
    return rows


def per_model(rows: dict, measure: str, readers: tuple[str, ...], groups=("pooled 50/50", "real"), keep=None) -> list:
    """For each reader, per model, (flags of the first group, flags of the second), on units every
    reader labeled (and that keep accepts)."""
    out = {r: [] for r in readers}
    for model in MODELS:
        units = [u for u in rows[model] if all(u[measure][r] is not None for r in readers) and (keep is None or keep(u))]
        for r in readers:
            out[r].append(tuple(np.array([u[measure][r] for u in units if u["group"] == g], bool) for g in groups))
    return [out[r] for r in readers]


def difference(rows: dict, measure: str, reader: str, rng, groups=("pooled 50/50", "real"), keep=None) -> dict:
    """The mean over models of (first group's rate - second's) for one reader, with a bootstrap interval."""
    (pairs,) = per_model(rows, measure, (reader,), groups, keep)
    _, _, interval, _ = paired_gain(pairs, pairs, rng)
    return {"mean_over_models": round(excess(pairs), 3), "interval": [round(x, 3) for x in interval],
            "per_model": {m: round(float(a.mean() - b.mean()), 3) for m, (a, b) in zip(MODELS, pairs)},
            "units": int(sum(len(a) + len(b) for a, b in pairs))}


def gain(rows: dict, measure: str, a: str, b: str, rng) -> dict:
    first, second = per_model(rows, measure, (a, b))
    point, interval, a_interval, b_interval = paired_gain(first, second, rng)
    return {"gain": round(point, 3), "interval": [round(x, 3) for x in interval], "verdict": gain_verdict(point, interval),
            f"{a} excess": [round(excess(first), 3), [round(x, 3) for x in a_interval]],
            f"{b} excess": [round(excess(second), 3), [round(x, 3) for x in b_interval]],
            "units": int(sum(len(p) + len(q) for p, q in first))}


def one_job(rows: dict, rng, group: str = "real") -> dict:
    """The share of a group's units (real neurons, in the rubric) each language-model reader calls one
    job, and each agent's paired difference from one-shot Haiku."""
    real = [u for model_rows in rows.values() for u in model_rows if u["group"] == group]
    out = {}
    for r in LLM_READERS:
        labeled = [u for u in real if u["labels"][r] is not None]
        share = np.array([u["labels"][r] == ONE_JOB for u in labeled])
        low, high = bootstrap_interval(share, rng)
        out[r] = {"share": round(float(share.mean()), 3), "interval": [round(low, 3), round(high, 3)],
                  "per_model": {m: round(float(np.mean([u["labels"][r] == ONE_JOB for u in labeled if u["model"] == m])), 3)
                                for m in MODELS}}
    for agent in AGENTS:
        both = [u for u in real if u["labels"][agent] is not None and u["labels"]["haiku"] is not None]
        d = np.array([(u["labels"][agent] == ONE_JOB) - (u["labels"]["haiku"] == ONE_JOB) for u in both], float)
        draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)]
        out[f"{agent} minus haiku"] = {"difference": round(float(d.mean()), 3),
                                       "interval": [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])]}
    return out


def specificity(rows: dict, reader: str, rng) -> dict:
    no_shared = lambda u: not u["shared_window"]  # noqa: E731
    return {"pooled minus related": difference(rows, "flag", reader, rng, ("pooled 50/50", "related")),
            "related minus real": difference(rows, "flag", reader, rng, ("related", "real")),
            "related minus real, no shared window": difference(rows, "flag", reader, rng, ("related", "real"), no_shared)}


def auc(pooled: np.ndarray, real: np.ndarray) -> float:
    """The chance a random pool scores above a random real neuron, ties counting half."""
    diff = pooled[:, None] - real[None, :]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))


def label_auc(rows: dict, reader: str, rng) -> dict:
    """Post hoc: how well a reader's labels (A < B < C, or a detector's flag) rank pools above real
    neurons, whatever its threshold: the mean over models of the AUC, with a bootstrap interval."""
    def score(u):
        return LABEL_RANK.get(u["labels"][reader]) if reader in LLM_READERS else int(u["flag"][reader])

    per_model = []
    for model in MODELS:
        units = [u for u in rows[model] if score(u) is not None]
        per_model.append(tuple(np.array([score(u) for u in units if u["group"] == g], float) for g in ("pooled 50/50", "real")))
    draws = [np.mean([auc(p[rng.integers(0, len(p), len(p))], r[rng.integers(0, len(r), len(r))]) for p, r in per_model])
             for _ in range(BOOTSTRAP)]
    return {"mean_over_models": round(float(np.mean([auc(p, r) for p, r in per_model])), 3),
            "interval": [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])]}


def catch_all(rows: dict, patterns: dict) -> dict:
    """Post hoc: the share of one-job (A) answers whose stated pattern uses a catch-all word such as
    "predictable" or "formulaic", for real neurons and for pools. A crude keyword count."""
    out = {}
    for reader in LLM_READERS:
        out[reader] = {}
        for g in ("real", "pooled 50/50"):
            ones = [u["id"] for m in MODELS for u in rows[m] if u["group"] == g and u["labels"][reader] == ONE_JOB]
            hits = [any(w in " ".join(patterns[reader][k]).lower() for w in CATCH_ALL_WORDS) for k in ones]
            out[reader][g] = {"one-job answers": len(ones), "share with a catch-all word": round(float(np.mean(hits)), 3)}
    return out


def kappas(rows: dict) -> dict:
    """Cohen's kappa on real neurons between each agent and every other reader, over all models and per model."""
    out = {}
    for agent in AGENTS:
        for other in READERS:
            if other == agent or (other in AGENTS and other < agent):
                continue
            pairs = {m: [(u["flag"][agent], u["flag"][other]) for u in rows[m] if u["group"] == "real"
                         and u["flag"][agent] is not None and u["flag"][other] is not None] for m in MODELS}
            everything = [p for m in MODELS for p in pairs[m]]
            out[f"{agent} vs {other}"] = {
                "all models": round(cohen_kappa(*np.array(everything).T), 3),
                **{m: round(cohen_kappa(*np.array(pairs[m]).T), 3) for m in MODELS}}
    return out


def retest(answers: dict, rng) -> dict:
    out = {}
    for agent, arm in AGENTS.items():
        pairs = [(answers[arm]["first"][k]["label"], v["label"], k) for k, v in answers[arm]["retest"].items()]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        first, second = (np.array([p[i] == UNRELATED_JOBS for p in pairs]) for i in (0, 1))
        draws = [cohen_kappa(first[idx], second[idx]) for idx in (rng.integers(0, len(pairs), len(pairs))
                                                                    for _ in range(BOOTSTRAP))]
        out[agent] = {"units": len(pairs), "real units": sum(":real:" in k for *_, k in pairs),
                      "same_label": round(float(np.mean([a == b for a, b, _ in pairs])), 3),
                      "same_flag": round(float(np.mean(first == second)), 3),
                      "flag_kappa": round(cohen_kappa(first, second), 3),
                      "flag_kappa_interval": [round(float(x), 3) for x in np.nanpercentile(draws, [2.5, 97.5])]}
    return out


def rates(rows: dict) -> dict:
    out = {}
    for i, model in enumerate(MODELS):
        out[model] = {}
        for j, r in enumerate(READERS):
            out[model][r] = {}
            for g in GROUPS:
                flags = np.array([u["flag"][r] for u in rows[model] if u["group"] == g and u["flag"][r] is not None], bool)
                low, high = bootstrap_interval(flags, rng_for(model, 40 + j))
                out[model][r][g] = {"share": round(float(flags.mean()), 3), "interval": [round(low, 3), round(high, 3)],
                                    "n": len(flags)}
    return out


def readings_of(answers: dict, arm: str) -> list[dict]:
    return [r for part in answers[arm].values() for r in part.values()]


def exploratory(rows: dict, answers: dict) -> dict:
    """Listed in the rubric before the run: tool use, confidence, and the cost of the readings."""
    group_of = {u["id"]: u["group"] for model_rows in rows.values() for u in model_rows}
    out = {}
    for agent, arm in AGENTS.items():
        first = answers[arm]["first"]
        tools = {}
        for g in GROUPS:
            records = [first[k] for k in first if group_of[k] == g]
            calls = [r["tool_calls"] for r in records]
            tools[g] = {
                "mean tool calls": round(float(np.mean([len(c) for c in calls])), 2),
                f"share with at least {MIN_TOOL_CALLS}": round(float(np.mean([len(c) >= MIN_TOOL_CALLS for c in calls])), 3),
                "share writing test text": round(float(np.mean([any(x["tool"] == "activation" for x in c) for c in calls])), 3),
                "share reading past rank 50": round(float(np.mean([any(x["tool"] == "examples" and int(
                    x["arguments"].get("start", 1)) + int(x["arguments"].get("count", 1)) > 51 for x in c) for c in calls])), 3),
                "mean test texts": round(float(np.mean([sum(len(x["arguments"].get("texts") or []) for x in c
                                                            if x["tool"] == "activation") for c in calls])), 2)}
        by_confidence = {}
        for g in ("real", "pooled 50/50"):
            for level in ("low", "medium", "high"):
                labels = [first[k]["label"] for k in first if group_of[k] == g and first[k]["confidence"] == level]
                if labels:
                    by_confidence[f"{g}, {level}"] = {"n": len(labels),
                                                      "flag rate": round(float(np.mean([x == UNRELATED_JOBS for x in labels])), 3)}
        records = readings_of(answers, arm)
        out[agent] = {"tool use": tools, "flag rate by confidence": by_confidence,
                      "labels": {g: dict(Counter(first[k]["label"] for k in first if group_of[k] == g)) for g in GROUPS},
                      "readings": len(records), "without a label": sum(r["label"] is None for r in records),
                      "with retries": sum(len(r["attempts"]) > 1 for r in records),
                      "all offered both tools": all(len(r["tools_offered"]) == 2 for r in records if r["label"]),
                      "models used": sorted({m for r in records for m in r["models"]}),
                      "cost_usd": round(sum(r["cost_usd"] for r in records), 2),
                      "input_tokens": sum(r["input_tokens"] for r in records),
                      "output_tokens": sum(r["output_tokens"] for r in records)}
    both = [u for model_rows in rows.values() for u in model_rows
            if u["flag"]["haiku agent"] is not None and u["flag"]["sonnet agent"] is not None]
    out["agents agree, kappa on all units"] = round(cohen_kappa(*np.array([(u["flag"]["haiku agent"], u["flag"]["sonnet agent"])
                                                                         for u in both]).T), 3)
    return out


def tool_summary(call: dict) -> str:
    args = call["arguments"]
    if call["tool"] == "examples":
        start = int(args.get("start", 1))
        return f"examples {start}-{start + int(args.get('count', 1)) - 1}"
    strongest = [line.split(" at ")[0].split("strongest ")[-1] for line in (call["answer"] or "").split("\n")
                 if line.startswith("Text ")]
    texts = "; ".join(f"\"{t[:60]}\" ({s})" for t, s in zip(args.get("texts") or [], strongest))
    return f"activation on {texts}"


def write_report(rows: dict, stats: dict, answers: dict) -> None:
    lines = ["# v8 results per model\n",
             "Flag rates (unrelated jobs) by reader and unit type at the middle layer, then, per model, up to "
             f"{EXAMPLES_SHOWN} units of each kind worth reading: 50/50 pools an agent called one job, and real neurons "
             "the agents disagree on. For each, every reader's label, the agents' stated patterns and their tool calls "
             "(for test texts, the unit's strongest activation on each, on the 0-10 scale). Unit ids use v6's sample "
             "indices; the MLP neuron ids are given beside them.\n"]
    for model in MODELS:
        lines.append(f"## pythia-{model}\n")
        lines.append("| Reader | " + " | ".join(GROUPS) + " |\n|---|" + "---|" * len(GROUPS))
        for r in READERS:
            lines.append(f"| {r} | " + " | ".join(f"{stats['rates'][model][r][g]['share']:.0%}" for g in GROUPS) + " |")
        lines.append("")
        model_rows = rows[model]
        for title, chosen in (
            ("50/50 pools the Haiku agent calls one job",
             [u for u in model_rows if u["group"] == "pooled 50/50" and u["labels"]["haiku agent"] == ONE_JOB]),
            ("50/50 pools the Sonnet agent calls one job",
             [u for u in model_rows if u["group"] == "pooled 50/50" and u["labels"]["sonnet agent"] == ONE_JOB]),
            ("Real neurons the agents disagree on",
             [u for u in model_rows if u["group"] == "real" and u["flag"]["haiku agent"] is not None
              and u["flag"]["sonnet agent"] is not None and u["flag"]["haiku agent"] != u["flag"]["sonnet agent"]])):
            lines.append(f"### {title}\n")
            for u in chosen[:EXAMPLES_SHOWN]:
                lines.append(f"**{u['id']}**, MLP neuron(s) {', '.join(map(str, u['neuron_ids']))}: "
                             + ", ".join(f"{r} {u['labels'][r] or '-'}" for r in LLM_READERS)
                             + f", network {u['flag']['network']}, MiniLM {u['flag']['minilm']}\n")
                for agent, arm in AGENTS.items():
                    record = answers[arm]["first"].get(u["id"])
                    if record and record["label"]:
                        lines.append(f"- {agent} ({record['label']}, {record['confidence']} confidence): "
                                     + " / ".join(record["patterns"])[:600])
                        lines += [f"  - {tool_summary(c)}" for c in record["tool_calls"]]
                lines.append("")
    (RESULTS_DIR / "v8_models.md").write_text("\n".join(lines))


def main() -> None:
    answers = json.loads(ANSWERS_PATH.read_text())
    v7 = json.loads((RESULTS_DIR / "classification_v7.json").read_text())
    v7_replies = json.loads((RESULTS_DIR / "v7_llm_answers.json").read_text())["first"]
    rows = unit_rows(answers, v7["units"])
    unit_ids = [u["id"] for m in MODELS for u in rows[m]]
    patterns = {"haiku": {k: (parse_answer(v7_replies[k]["reply"]) or {"patterns": []})["patterns"] for k in unit_ids},
                **{name: {k: answers[arm]["first"][k]["patterns"] for k in unit_ids} for name, arm in AGENTS.items()}}
    rng = np.random.default_rng([8, 99])
    stats = {
        "pooled_excess": {measure: {r: difference(rows, measure, r, rng) for r in (READERS if measure == "flag" else LLM_READERS)}
                          for measure in MEASURES},
        "gains": {measure: {f"{a} over {b}": gain(rows, measure, a, b, rng) for a, b in GAINS} for measure in MEASURES},
        "one_job_real": one_job(rows, rng),
        "one_job_pools": one_job(rows, rng, "pooled 50/50"),  # exploratory: how often a known pool is missed
        "specificity": {r: specificity(rows, r, rng) for r in READERS},
        "kappa_real": kappas(rows),
        "retest": retest(answers, rng),
        "rates": rates(rows),
        "exploratory": exploratory(rows, answers),
        "post_hoc": {"label_auc": {r: label_auc(rows, r, rng) for r in READERS},
                     "catch_all_one_job": catch_all(rows, patterns)},
    }
    stats["verdict"] = stats["gains"]["flag"][f"{PRIMARY[0]} over {PRIMARY[1]}"]["verdict"]
    units_out = {m: [{k: v for k, v in u.items()} for u in rows[m]] for m in MODELS}
    (RESULTS_DIR / "classification_v8.json").write_text(json.dumps({"summary": stats, "units": units_out}, indent=1))
    write_report(rows, stats, answers)
    print(json.dumps({k: stats[k] for k in ("verdict", "gains", "pooled_excess", "one_job_real", "retest")}, indent=1))


if __name__ == "__main__":
    main()
