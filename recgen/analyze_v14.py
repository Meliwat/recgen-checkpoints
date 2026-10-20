"""Apply the pre-registered v14 rules (recgen/rubric_v14.py): the Opus agent beside v8's two agents
and v7's three readers, on the same units, with v8's machinery (recgen/analyze_v8).

Reads results files only: the Opus agent's readings (results/v14_agent_answers.json, from
recgen.agent_v14), v8's agents' readings (results/v8_agent_answers.json), and v7's per-unit flags,
labels and replies (results/classification_v7.json, results/v7_llm_answers.json). Writes
results/classification_v14.json. A unit a reader never labeled is left out of that reader's rates
and of the paired comparisons that use it, and counted.
"""

import json

import numpy as np

from recgen import analyze_v8
from recgen.agent_v8 import ANSWERS_PATH as V8_ANSWERS_PATH
from recgen.agent_v14 import ANSWERS_PATH
from recgen.analyze_v8 import (
    MEASURES, agent_use, catch_all, difference, gain, kappas, label_auc, one_job, one_job_difference, rates, retest,
    specificity, unit_rows,
)
from recgen.config import RESULTS_DIR
from recgen.rubric_v8 import MODELS, parse_answer
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
    }
    stats["verdict"] = stats["gains"]["flag"][f"{PRIMARY[0]} over {PRIMARY[1]}"]["verdict"]
    (RESULTS_DIR / "classification_v14.json").write_text(json.dumps({"summary": stats, "units": rows}, indent=1))
    print(json.dumps({k: stats[k] for k in ("verdict", "gains", "prompts_match_v8")}, indent=1))


if __name__ == "__main__":
    main()
