"""Apply the pre-registered v14 rules (recgen/rubric_v14.py): the Opus agent beside v8's two agents
and v7's three readers, on the same units, with v8's machinery (recgen/analyze_v8).

Reads results files only: the Opus agent's readings (results/v14_agent_answers.json, from
recgen.agent_v14), v8's agents' readings (results/v8_agent_answers.json), and v7's per-unit flags,
labels and replies (results/classification_v7.json, results/v7_llm_answers.json). Writes
results/classification_v14.json. A unit a reader never labeled is left out of that reader's rates
and of the paired comparisons that use it, and counted.
"""

import json
from collections import Counter

import numpy as np

from recgen import analyze_v8
from recgen.agent_v8 import ANSWERS_PATH as V8_ANSWERS_PATH
from recgen.agent_v14 import ANSWERS_PATH
from recgen.analyze_v8 import (
    LABEL_RANK, MEASURES, agent_use, auc, catch_all, difference, gain, kappas, label_auc, one_job, one_job_difference,
    rates, readings_of, retest, specificity, unit_rows,
)
from recgen.config import RESULTS_DIR
from recgen.rubric_v7 import ONE_JOB, UNRELATED_JOBS
from recgen.rubric_v8 import BOOTSTRAP, MAX_TOOL_CALLS, MODELS, parse_answer
from recgen.rubric_v14 import AGENT_MODEL, AGENT_NAME, PRIMARY, SECONDARY

AGENTS = {**analyze_v8.AGENTS, AGENT_NAME: AGENT_MODEL}
LLM_READERS = ("haiku", *AGENTS)  # "haiku" is v7's one-shot reader
READERS = ("minilm", "network") + LLM_READERS
GAINS = (PRIMARY, *SECONDARY)


def load_answers() -> dict:
    """v8's two agents' readings and the Opus agent's, keyed by the agent's model as in v8."""
    return {**json.loads(V8_ANSWERS_PATH.read_text()), **json.loads(ANSWERS_PATH.read_text())}


def prompts_match(answers: dict) -> dict:
    """How many Opus readings saw exactly the prompt v8's Sonnet agent saw for the same unit."""
    opus, sonnet = answers[AGENT_MODEL]["first"], answers[analyze_v8.AGENTS["sonnet agent"]]["first"]
    same = [opus[k]["prompt_sha256"] == sonnet[k]["prompt_sha256"] for k in opus if k in sonnet]
    return {"same": int(sum(same)), "compared": len(same), "opus readings": len(opus)}


def one_job_shares(rows: dict, group: str, rng) -> dict:
    """Every language-model reader's one-job share of a group, each agent's paired difference from
    one-shot Haiku, and the primary pair's paired difference."""
    out = one_job(rows, rng, group, AGENTS)
    units = [u for model_rows in rows.values() for u in model_rows if u["group"] == group]
    out[f"{PRIMARY[0]} minus {PRIMARY[1]}"] = one_job_difference(units, *PRIMARY, rng)
    return out


def tool_use(rows: dict, answers: dict) -> dict:
    group_of = {u["id"]: u["group"] for model_rows in rows.values() for u in model_rows}
    out = {}
    for agent, arm in AGENTS.items():
        out[agent] = agent_use(group_of, answers, arm)
        first = answers[arm]["first"].values()
        out[agent]["mean tool calls, all first readings"] = round(float(np.mean([len(r["tool_calls"]) for r in first])), 2)
        out[agent]["most tool calls"] = max(len(r["tool_calls"]) for r in first)
        # Post hoc, added after seeing the results: whether the budget bound.
        out[agent][f"share using all {MAX_TOOL_CALLS}"] = round(float(np.mean([len(r["tool_calls"]) >= MAX_TOOL_CALLS
                                                                                for r in first])), 3)
    return out


def label_changes(rows: dict, a: str, b: str) -> dict:
    """Post hoc, added after seeing the results: per unit group, how many units each pair of labels
    (reader a's, then reader b's) covers, as "A->B" counts."""
    out = {}
    for group in ("real", "pooled 50/50"):
        pairs = Counter(f"{u['labels'][a]}->{u['labels'][b]}" for model_rows in rows.values() for u in model_rows
                        if u["group"] == group and u["labels"][a] is not None and u["labels"][b] is not None)
        out[group] = dict(sorted(pairs.items()))
    return out


def label_shares(rows: dict, readers: tuple[str, ...] = LLM_READERS) -> dict:
    """Post hoc, added after an independent review: each language-model reader's shares of A, B and C on
    real neurons and on 50/50 pools. "B or C" is the secondary flag; "A or B" (1 minus the C share) is
    the exploratory reading of the essay's "one main job". The B-or-C ceiling is the largest B-or-C
    detection a reader could reach given how often it calls real neurons B or C: the mean over sizes of
    1 minus that rate."""
    out = {}
    for r in readers:
        out[r] = {}
        for g in ("real", "pooled 50/50"):
            labels = {m: [u["labels"][r] for u in rows[m] if u["group"] == g and u["labels"][r] is not None] for m in MODELS}
            every = [x for m in MODELS for x in labels[m]]
            out[r][g] = {**{x: round(every.count(x) / len(every), 3) for x in LABEL_RANK},
                         "B or C": round(float(np.mean([x != ONE_JOB for x in every])), 3),
                         "A or B": round(float(np.mean([x != UNRELATED_JOBS for x in every])), 3),
                         "B or C per model": {m: round(float(np.mean([x != ONE_JOB for x in labels[m]])), 3) for m in MODELS}}
        real_rates = [np.mean([u["labels"][r] != ONE_JOB for u in rows[m] if u["group"] == "real" and u["labels"][r] is not None])
                      for m in MODELS]
        out[r]["B-or-C ceiling"] = round(float(np.mean([1 - x for x in real_rates])), 3)
    return out


def label_auc_difference(rows: dict, a: str, b: str, rng) -> dict:
    """Post hoc, added after an independent review: reader a's label AUC (50/50 pools over real neurons,
    mean over sizes) minus reader b's, on the units both labeled, with v8's paired resampling: units
    drawn within size and unit type, both readers' labels moving together."""
    per_model = []
    for model in MODELS:
        units = [u for u in rows[model] if u["labels"][a] is not None and u["labels"][b] is not None]
        per_model.append(tuple(np.array([[LABEL_RANK[u["labels"][r]] for r in (a, b)] for u in units if u["group"] == g], float)
                               for g in ("pooled 50/50", "real")))

    def difference_of(samples) -> float:
        return float(np.mean([auc(p[:, 0], r[:, 0]) - auc(p[:, 1], r[:, 1]) for p, r in samples]))

    draws = [difference_of([(p[rng.integers(0, len(p), len(p))], r[rng.integers(0, len(r), len(r))]) for p, r in per_model])
             for _ in range(BOOTSTRAP)]
    return {"difference": round(difference_of(per_model), 3),
            "interval": [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])], "draws": BOOTSTRAP}


def shared_unrelated(rows: dict, a: str, b: str) -> dict:
    """Post hoc, added after an independent review: per unit group, how many units each reader calls C
    (unrelated jobs), and how many get C from both, on units both labeled."""
    groups = ("real", "pooled 50/50")
    out = {}
    for g in groups:
        units = [u for m in MODELS for u in rows[m]
                 if u["group"] == g and u["labels"][a] is not None and u["labels"][b] is not None]
        out[g] = {f"{a} C": sum(u["labels"][a] == UNRELATED_JOBS for u in units),
                  f"{b} C": sum(u["labels"][b] == UNRELATED_JOBS for u in units),
                  "both C": sum(u["labels"][a] == u["labels"][b] == UNRELATED_JOBS for u in units)}
    out["both groups"] = {k: sum(out[g][k] for g in groups) for k in out[groups[0]]}
    return out


def reading_stats(answers: dict) -> dict:
    """Post hoc, added after an independent review: per agent, over all its readings (retests included),
    the deepest rank of the ranked list it asked for, and its output tokens and CLI-reported cost per
    reading."""
    out = {}
    for agent, arm in AGENTS.items():
        records = readings_of(answers, arm)
        requests = [c["arguments"] for r in records for c in r["tool_calls"] if c["tool"] == "examples"]
        out[agent] = {"readings": len(records),
                      "deepest rank asked for": max((int(x.get("start", 1)) + int(x.get("count", 1)) - 1 for x in requests),
                                                    default=None),
                      "output tokens per reading": round(float(np.mean([r["output_tokens"] for r in records])), 1),
                      "cost per reading, USD": round(float(np.mean([r["cost_usd"] for r in records])), 4)}
    return out


def main() -> None:
    answers = load_answers()
    v7 = json.loads((RESULTS_DIR / "classification_v7.json").read_text())
    v7_replies = json.loads((RESULTS_DIR / "v7_llm_answers.json").read_text())["first"]
    rows = unit_rows(answers, v7["units"], AGENTS)
    unit_ids = [u["id"] for m in MODELS for u in rows[m]]
    patterns = {"haiku": {k: (parse_answer(v7_replies[k]["reply"]) or {"patterns": []})["patterns"] for k in unit_ids},
                **{name: {k: answers[arm]["first"][k]["patterns"] for k in unit_ids} for name, arm in AGENTS.items()}}
    rng = np.random.default_rng([14, 99])
    review_rng = np.random.default_rng([14, 100])  # its own stream, so the checks added after review leave the rest unchanged
    stats = {
        "gains": {measure: {f"{a} over {b}": gain(rows, measure, a, b, rng) for a, b in GAINS} for measure in MEASURES},
        "pooled_excess": {measure: {r: difference(rows, measure, r, rng) for r in (READERS if measure == "flag" else LLM_READERS)}
                          for measure in MEASURES},
        "label_auc": {r: label_auc(rows, r, rng) for r in READERS},
        "one_job_real": one_job_shares(rows, "real", rng),
        "one_job_pools": one_job_shares(rows, "pooled 50/50", rng),
        "retest": retest(answers, rng, AGENTS),
        "kappa_real": kappas(rows, {AGENT_NAME: AGENT_MODEL}, READERS),
        "specificity": {r: specificity(rows, r, rng) for r in READERS},
        "tool_use": tool_use(rows, answers),
        "rates": rates(rows, READERS),
        "catch_all_one_job": catch_all(rows, patterns),
        "prompts_match_v8": prompts_match(answers),
        "post_hoc": {"label changes, sonnet agent to opus agent": label_changes(rows, "sonnet agent", AGENT_NAME)},
        "post_hoc_after_review": {
            "note": "post hoc, added after an independent review",
            "label shares": label_shares(rows),
            f"label AUC, {PRIMARY[0]} minus {PRIMARY[1]}": label_auc_difference(rows, *PRIMARY, review_rng),
            f"C labels, {PRIMARY[0]} and {PRIMARY[1]}": shared_unrelated(rows, *PRIMARY),
            "per reading": reading_stats(answers),
        },
    }
    per_reading = stats["post_hoc_after_review"]["per reading"]
    stats["post_hoc_after_review"][f"cost per reading, {PRIMARY[0]} over {PRIMARY[1]}"] = round(
        per_reading[PRIMARY[0]]["cost per reading, USD"] / per_reading[PRIMARY[1]]["cost per reading, USD"], 2)
    stats["verdict"] = stats["gains"]["flag"][f"{PRIMARY[0]} over {PRIMARY[1]}"]["verdict"]
    (RESULTS_DIR / "classification_v14.json").write_text(json.dumps({"summary": stats, "units": rows}, indent=1))
    print(json.dumps({k: stats[k] for k in ("verdict", "gains", "prompts_match_v8")}, indent=1))


if __name__ == "__main__":
    main()
