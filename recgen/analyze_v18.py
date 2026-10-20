"""Apply the pre-registered v18 rules (recgen/rubric_v18.py): does the falsification-driven Opus agent's
label separate 50/50 pools from real neurons better than v14's Opus agent, and does its final
description predict held-out activations better than v15's one-shot explanation?

Reads results files only: v18's readings (results/v18_agent_answers.json, from recgen.agent_v18), v15's
held-out sets, explanations and simulations (results/v15_answers.json), v8's label AUCs
(results/classification_v8.json) and an extract of v14's Opus readings (results/v14_opus_extract.json,
made once, read-only, from the v14 branch's results/v14_agent_answers.json). A unit a reader never
labeled or scored is left out of that reader's numbers and counted.
"""

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from recgen.agent_v18 import ANSWERS_PATH
from recgen.analyze_v8 import CATCH_ALL_WORDS, auc, difference, gain
from recgen.analyze_v15 import compare, paired_mean, tracked_neurons
from recgen.config import RESULTS_DIR
from recgen.rubric_v5 import bootstrap_interval
from recgen.rubric_v6 import cohen_kappa
from recgen.rubric_v7 import ONE_JOB, UNRELATED_JOBS
from recgen.rubric_v8 import BOOTSTRAP, EXAMPLES_PER_CALL, MODELS, N_WINDOWS, NO_CALLS_LEFT, paired_gain
from recgen.rubric_v15 import PRIMARY_SIMULATOR, unit_scores, verdict as v15_verdict
from recgen.rubric_v18 import (
    AGENT_MODEL, LABEL_RANK, MAX_DESCRIPTION_WORDS, MAX_TOOL_CALLS, MIN_TOOL_CALLS, PROTOCOL_RANKS, auc_verdict, label_auc,
    paired_auc_difference,
)

V14_WORKTREE = Path.home() / "recgen-checkpoints-v14"
V14_SOURCE = V14_WORKTREE / "results" / "v14_agent_answers.json"
V14_EXTRACT_PATH = RESULTS_DIR / "v14_opus_extract.json"
V14_ARM = "claude-opus-5-5"
READERS = ("v18", "v14")  # the v18 agent and v14's Opus agent
ARMS = ("v18", "v15")  # the v18 description and v15's explanation, both scored by v15's Sonnet simulator
GROUPS = ("real", "pooled 50/50", "related")
V8_READERS = ("minilm", "network", "haiku", "haiku agent", "sonnet agent")
V15_AUC = 0.56  # v15's score AUC, real over pools, quoted in the rubric


def tool_use(calls: list[dict], held_out: list[int] | None = None) -> dict:
    """What one reading did with its tools. Example ranks are mapped to the unit's full ranked list,
    skipping held_out (the ranks the agent's list leaves out), so v14's and v18's ranks compare. Calls
    past the budget, which got no answer, read nothing."""
    held = set(held_out or [])
    kept = [r for r in range(1, N_WINDOWS + 1) if r not in held]
    ranks = []
    for call in calls:
        if call["tool"] != "examples" or call["answer"] == NO_CALLS_LEFT:
            continue
        try:
            start, count = max(1, int(call["arguments"]["start"])), max(1, min(int(call["arguments"]["count"]), EXAMPLES_PER_CALL))
        except (KeyError, TypeError, ValueError):
            continue
        ranks += [kept[r - 1] for r in range(start, start + count) if r <= len(kept)]
    low, high = PROTOCOL_RANKS
    return {"calls": len(calls), "test texts": sum(len(c["arguments"].get("texts") or []) for c in calls
                                                   if c["tool"] == "activation" and isinstance(c["arguments"], dict)),
            "deepest rank": max(ranks, default=None), "ranks read past 50": sum(r > 50 for r in ranks),
            f"read ranks {low}-{high}": any(low <= r <= high for r in ranks)}


def extract_v14(source: Path = V14_SOURCE) -> dict:
    """The parts of v14's Opus readings v18 compares with, from the v14 branch (read-only)."""
    raw = source.read_bytes()
    answers = json.loads(raw)[V14_ARM]
    commit = subprocess.run(["git", "-C", str(V14_WORKTREE), "log", "-1", "--format=%H", "--", "results/v14_agent_answers.json"],
                            capture_output=True, text=True).stdout.strip()
    return {"source": "results/v14_agent_answers.json on branch v14", "source_commit": commit,
            "source_sha256": hashlib.sha256(raw).hexdigest(), "agent": V14_ARM,
            **{reading: {k: {"label": r["label"], "confidence": r["confidence"], "prompt_sha256": r["prompt_sha256"],
                             "cost_usd": r["cost_usd"], "output_tokens": r["output_tokens"], **tool_use(r["tool_calls"])}
                         for k, r in answers[reading].items()} for reading in ("first", "retest")}}


def v14_extract() -> dict:
    if not V14_EXTRACT_PATH.exists():
        V14_EXTRACT_PATH.write_text(json.dumps(extract_v14(), indent=1))
    return json.loads(V14_EXTRACT_PATH.read_text())


def unit_rows(answers: dict, v15: dict, v14: dict) -> dict[str, list[dict]]:
    """Each unit of each size with both readers' labels and flags and both arms' scores, in the shapes
    analyze_v8 and analyze_v15 read (flag/not_one_job by reader; first/scores by arm)."""
    rows = {model: [] for model in MODELS}
    for unit_id, u in answers["units"].items():
        truth = v15["units"][unit_id]["held_out"]
        values, strata = [h["value"] for h in truth], [h["stratum"] for h in truth]
        agent = u["first"]["agent"]
        labels = {"v18": agent["label"], "v14": v14["first"][unit_id]["label"]}
        predictions = {"v18": (u["first"]["simulation"] or {}).get("parsed"),
                       "v15": v15["units"][unit_id]["first"]["simulations"][PRIMARY_SIMULATOR]["parsed"]}
        retest = u.get("retest")
        rows[u["model"]].append({
            "id": unit_id, "model": u["model"], "group": u["group"], "labels": labels,
            "flag": {r: None if x is None else x == UNRELATED_JOBS for r, x in labels.items()},
            "not_one_job": {r: None if x is None else x != ONE_JOB for r, x in labels.items()},
            "confidence": agent["confidence"], "description": agent.get("description"),
            "explanation": v15["units"][unit_id]["first"]["explanation"]["parsed"],
            "judge": {"v18": (u["first"]["judge"] or {}).get("parsed"),
                      "v15": (v15["units"][unit_id]["first"].get("judge") or {}).get("parsed")},
            "first": {"scores": {a: None if p is None else unit_scores(p, values, strata) for a, p in predictions.items()}},
            "retest": None if retest is None else {
                "labels": {"v18": retest["agent"]["label"], "v14": v14["retest"].get(unit_id, {}).get("label")},
                "score": None if not (retest["simulation"] or {}).get("parsed") else unit_scores(
                    retest["simulation"]["parsed"], values, strata)["pearson"]},
            "tool use": tool_use(agent["tool_calls"], u["held_out_ranks"]), "v14 tool use": {
                k: v14["first"][unit_id][k] for k in ("calls", "test texts", "deepest rank", f"read ranks {PROTOCOL_RANKS[0]}-{PROTOCOL_RANKS[1]}")},
            "cost_usd": agent["cost_usd"], "v14 cost_usd": v14["first"][unit_id]["cost_usd"]})
    return rows


def label_ranks(rows: dict, readers: tuple[str, ...], first_group: str = "pooled 50/50", second_group: str = "real") -> list:
    """For each reader, per size, (label ranks of the first group, of the second), on units every reader labeled."""
    out = {r: [] for r in readers}
    for model in MODELS:
        units = [u for u in rows[model] if all(u["labels"][r] is not None for r in readers)]
        for r in readers:
            out[r].append(tuple(np.array([LABEL_RANK[u["labels"][r]] for u in units if u["group"] == g], float)
                                for g in (first_group, second_group)))
    return [out[r] for r in readers]


def primary(rows: dict, rng) -> dict:
    """Label AUC (A < B < C) for ranking 50/50 pools above real neurons, v18 minus v14, paired."""
    first, second = label_ranks(rows, READERS)
    point, interval, a_interval, b_interval = paired_auc_difference(first, second, rng)
    return {"v18 minus v14": round(point, 3), "interval": [round(x, 3) for x in interval],
            "verdict": auc_verdict(point, interval),
            "v18 label auc": [round(label_auc(first), 3), [round(x, 3) for x in a_interval]],
            "v14 label auc": [round(label_auc(second), 3), [round(x, 3) for x in b_interval]],
            "per_model": {m: {"v18": round(auc(*a), 3), "v14": round(auc(*b), 3)} for m, a, b in zip(MODELS, first, second)},
            "units": int(sum(len(p) + len(r) for p, r in first))}


def one_job(rows: dict, rng) -> dict:
    """The share each reader calls one job (A), for real neurons and 50/50 pools, and v18 minus v14 on real neurons."""
    out = {}
    for g in ("real", "pooled 50/50"):
        units = [u for m in MODELS for u in rows[m] if u["group"] == g]
        out[g] = {}
        for r in READERS:
            share = np.array([u["labels"][r] == ONE_JOB for u in units if u["labels"][r] is not None])
            low, high = bootstrap_interval(share, rng)
            out[g][r] = {"share": round(float(share.mean()), 3), "interval": [round(low, 3), round(high, 3)], "n": len(share)}
    real = [u for m in MODELS for u in rows[m] if u["group"] == "real" and None not in u["labels"].values()]
    d = np.array([(u["labels"]["v18"] == ONE_JOB) - (u["labels"]["v14"] == ONE_JOB) for u in real], float)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)]
    out["real, v18 minus v14"] = {"difference": round(float(d.mean()), 3),
                                  "interval": [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])]}
    return out


def label_shares(rows: dict) -> dict:
    """Descriptive: each reader's A/B/C shares by unit type and size."""
    out = {}
    for r in READERS:
        out[r] = {}
        for g in GROUPS:
            for model in MODELS + ["all sizes"]:
                labels = [u["labels"][r] for m in MODELS for u in rows[m] if u["group"] == g and (model in (m, "all sizes"))
                          and u["labels"][r] is not None]
                out[r][f"{g}, {model}"] = {x: round(labels.count(x) / len(labels), 3) for x in LABEL_RANK}
    return out


def score_tests(rows: dict, rng) -> dict:
    """Secondary 3 and 4: the v18 descriptions' score as a mixture detector, beside v15's explanations on
    the same units, and the paired score difference by unit type."""
    out = {}
    for arm in ARMS:
        test = compare(rows, arm, "pearson", rng)
        test["v15 verdict rule"] = v15_verdict(test["interval"], test["auc_interval"])
        out[arm] = test
    pairs = {arm: [] for arm in ARMS}
    for model in MODELS:
        both = [u for u in rows[model] if all(u["first"]["scores"][a] is not None for a in ARMS)]
        for arm in ARMS:
            pairs[arm].append(tuple(np.array([u["first"]["scores"][arm]["pearson"] for u in both if u["group"] == g])
                                    for g in ("real", "pooled 50/50")))
    point, interval, a_interval, b_interval = paired_auc_difference(pairs["v18"], pairs["v15"], rng)
    out["score auc, v18 minus v15"] = {"difference": round(point, 3), "interval": [round(x, 3) for x in interval],
                                       "v18": [round(label_auc(pairs["v18"]), 3), [round(x, 3) for x in a_interval]],
                                       "v15": [round(label_auc(pairs["v15"]), 3), [round(x, 3) for x in b_interval]],
                                       "units": int(sum(len(a) + len(b) for a, b in pairs["v18"]))}
    out["paired score difference, v18 minus v15"] = {}
    for g in GROUPS:
        per_model = [np.array([u["first"]["scores"]["v18"]["pearson"] - u["first"]["scores"]["v15"]["pearson"] for u in rows[m]
                               if u["group"] == g and all(u["first"]["scores"][a] is not None for a in ARMS)]) for m in MODELS]
        point, interval = paired_mean(per_model, rng)
        out["paired score difference, v18 minus v15"][g] = {
            "mean over sizes": round(point, 3), "interval": [round(x, 3) for x in interval],
            "per_model": {m: round(float(d.mean()), 3) for m, d in zip(MODELS, per_model)},
            "v18 mean": round(float(np.mean([np.mean([u["first"]["scores"]["v18"]["pearson"] for u in rows[m] if u["group"] == g
                                                       and all(u["first"]["scores"][a] is not None for a in ARMS)]) for m in MODELS])), 3),
            "v15 mean": round(float(np.mean([np.mean([u["first"]["scores"]["v15"]["pearson"] for u in rows[m] if u["group"] == g
                                                       and all(u["first"]["scores"][a] is not None for a in ARMS)]) for m in MODELS])), 3),
            "units": int(sum(map(len, per_model)))}
    return out


def retest(rows: dict, rng) -> dict:
    """Secondary 5: same label and C-flag kappa of each reader's two readings, on units both readers read twice."""
    units = [u for m in MODELS for u in rows[m] if u["retest"] is not None]
    out = {"units": len(units), "real units": sum(u["group"] == "real" for u in units)}
    for r in READERS:
        pairs = [(u["labels"][r], u["retest"]["labels"][r]) for u in units
                 if u["labels"][r] is not None and u["retest"]["labels"][r] is not None]
        first, second = (np.array([p[i] == UNRELATED_JOBS for p in pairs]) for i in (0, 1))
        draws = [cohen_kappa(first[idx], second[idx]) for idx in (rng.integers(0, len(pairs), len(pairs)) for _ in range(BOOTSTRAP))]
        out[r] = {"pairs": len(pairs), "same label": round(float(np.mean([a == b for a, b in pairs])), 3),
                  "same flag": round(float(np.mean(first == second)), 3), "flag kappa": round(cohen_kappa(first, second), 3),
                  "flag kappa interval": [round(float(x), 3) for x in np.nanpercentile(draws, [2.5, 97.5])]}
    scored = np.array([(u["first"]["scores"]["v18"]["pearson"], u["retest"]["score"]) for u in units
                       if u["first"]["scores"]["v18"] is not None and u["retest"]["score"] is not None])
    if len(scored) > 2:
        out["v18 simulation score, first vs retest (exploratory)"] = {
            "units": len(scored), "correlation": round(float(np.corrcoef(*scored.T)[0, 1]), 3),
            "mean absolute difference": round(float(np.mean(np.abs(scored[:, 0] - scored[:, 1]))), 3)}
    return out


def tools_and_cost(rows: dict, answers: dict) -> dict:
    """Secondary 6 and 7: tool use and cost, v18 beside v14 on the same units."""
    low, high = PROTOCOL_RANKS
    out = {}
    for name, key in (("v18", "tool use"), ("v14", "v14 tool use")):
        out[name] = {}
        for g in GROUPS + ("all",):
            uses = [u[key] for m in MODELS for u in rows[m] if g in ("all", u["group"])]
            depths = [x["deepest rank"] for x in uses if x["deepest rank"] is not None]
            out[name][g] = {"readings": len(uses), "mean calls": round(float(np.mean([x["calls"] for x in uses])), 2),
                            f"share with at least {MIN_TOOL_CALLS}": round(float(np.mean([x["calls"] >= MIN_TOOL_CALLS for x in uses])), 3),
                            f"share using all {MAX_TOOL_CALLS}": round(float(np.mean([x["calls"] >= MAX_TOOL_CALLS for x in uses])), 3),
                            "mean test texts": round(float(np.mean([x["test texts"] for x in uses])), 2),
                            "median deepest rank": None if not depths else int(np.median(depths)),
                            "largest deepest rank": None if not depths else int(max(depths)),
                            f"share reading ranks {low}-{high}": round(float(np.mean([x[f"read ranks {low}-{high}"] for x in uses])), 3)}
        out[name]["calls histogram"] = dict(sorted(Counter(u[key]["calls"] for m in MODELS for u in rows[m]).items()))
    records = {"agent, first": [u["first"]["agent"] for u in answers["units"].values()],
               "agent, retest": [u["retest"]["agent"] for u in answers["units"].values() if "retest" in u],
               "simulator, first": [u["first"]["simulation"] for u in answers["units"].values() if u["first"]["simulation"]],
               "simulator, retest": [u["retest"]["simulation"] for u in answers["units"].values()
                                     if "retest" in u and u["retest"]["simulation"]],
               "judge": [u["first"]["judge"] for u in answers["units"].values() if u["first"]["judge"]]}
    out["readings"] = {role: {"readings": len(rs), "without a valid reply": sum((r.get("label") if "label" in r else r["parsed"]) is None
                                                                                for r in rs),
                              "with retries": sum(len(r["attempts"]) > 1 for r in rs),
                              "stopped by the spend guard": sum(r["error"] == "spend cap reached" for r in rs),
                              "models used": sorted({m for r in rs for m in r["models"]}),
                              "cost_usd": round(sum(r["cost_usd"] for r in rs), 2),
                              "mean cost_usd": round(float(np.mean([r["cost_usd"] for r in rs])), 3) if rs else None,
                              "input_tokens": sum(r["input_tokens"] for r in rs), "output_tokens": sum(r["output_tokens"] for r in rs)}
                       for role, rs in records.items()}
    out["readings"]["v14 agent, first (same units)"] = {
        "mean cost_usd": round(float(np.mean([u["v14 cost_usd"] for m in MODELS for u in rows[m]])), 3)}
    out["total cost_usd (these readings)"] = round(sum(r["cost_usd"] for r in out["readings"].values() if "cost_usd" in r), 2)
    return out


def held_out_leak(answers: dict, v15: dict) -> dict:
    """Sanity check: held-out snippets whose text appears in any examples answer the agent got. The agent's
    list leaves out the held-out windows, so a match is the same text in another window; each match's
    shown rank is mapped back to the full list (post hoc detail, added after seeing three matches)."""
    matches = []
    for unit_id, u in answers["units"].items():
        hidden = {h["snippet"]: h for h in v15["units"][unit_id]["held_out"]}
        held = set(u["held_out_ranks"])
        kept = [r for r in range(1, N_WINDOWS + 1) if r not in held]
        for reading in ("first", "retest"):
            for call in (u.get(reading) or {}).get("agent", {}).get("tool_calls", []):
                if call["tool"] != "examples" or not call["answer"]:
                    continue
                for line in call["answer"].split("\n"):
                    snippet = line.split(": ", 1)[-1]
                    if line.startswith("rank ") and snippet in hidden:
                        shown = int(line.split()[1])
                        matches.append({"unit": unit_id, "reading": reading, "shown at full-list rank": kept[shown - 1],
                                        "shown value": float(line.split("(")[1].split(")")[0]),
                                        "held-out rank": hidden[snippet]["rank"],
                                        "held-out value": round(hidden[snippet]["value"], 2)})
    return {"held-out snippets shown to the agent": len(matches), "matches": matches}


def post_hoc(rows: dict, answers: dict, rng) -> dict:
    """Post hoc, added after seeing the results. (1) How each unit's label moved from v14's Opus agent to
    the v18 agent. (2) The primary test without readings whose CLI usage named a second model. (3) v8's
    catch-all keyword count on one-job descriptions (v8's words, unchanged)."""
    transitions = {g: dict(Counter(f"{u['labels']['v14']} -> {u['labels']['v18']}" for m in MODELS for u in rows[m]
                                   if u["group"] == g)) for g in ("real", "pooled 50/50")}
    mixed = sorted(k for k, u in answers["units"].items() for r in ("first", "retest")
                   if r in u and u[r]["agent"]["models"] != [AGENT_MODEL])
    kept = {m: [u for u in rows[m] if u["id"] not in mixed] for m in MODELS}
    first, second = label_ranks(kept, READERS)
    point, interval, _, _ = paired_auc_difference(first, second, rng)
    catch_all = {}
    for g in ("real", "pooled 50/50"):
        ones = [u["description"] for m in MODELS for u in rows[m] if u["group"] == g and u["labels"]["v18"] == ONE_JOB]
        catch_all[g] = {"one-job answers": len(ones), "share with a catch-all word": round(float(np.mean(
            [any(w in d.lower() for w in CATCH_ALL_WORDS) for d in ones])), 3)}
    return {"label transitions, v14 -> v18": transitions,
            "readings whose usage names a second model": mixed,
            "primary without them": {"v18 minus v14": round(point, 3), "interval": [round(x, 3) for x in interval],
                                     "verdict": auc_verdict(point, interval), "units": int(sum(len(p) + len(r) for p, r in first))},
            "catch-all words in one-job descriptions": catch_all}


def exploratory(rows: dict, answers: dict, v15: dict, rng) -> dict:
    """Listed in the rubric before the run."""
    synthetic = {"units": {}}
    for unit_id, u in answers["units"].items():
        synthetic["units"][unit_id] = {"held_out": v15["units"][unit_id]["held_out"], "first": {"simulations": {
            "v18": u["first"]["simulation"],
            "v15": v15["units"][unit_id]["first"]["simulations"][PRIMARY_SIMULATOR]}}}
    tracked = {arm: {g: tracked_neurons(rows, synthetic, arm, rng, g) for g in ("pooled 50/50", "related")} for arm in ARMS}
    judge = {}
    for g in GROUPS:
        units = [u for m in MODELS for u in rows[m] if u["group"] == g]
        judge[g] = {arm: round(float(np.mean([u["judge"][arm] for u in units if u["judge"][arm] is not None])), 3) for arm in ARMS}
        judge[g]["v18 judged"] = sum(u["judge"]["v18"] is not None for u in units)
    by_label = {}
    for g in ("real", "pooled 50/50"):
        for label in LABEL_RANK:
            scores = [u["first"]["scores"]["v18"]["pearson"] for m in MODELS for u in rows[m]
                      if u["group"] == g and u["labels"]["v18"] == label and u["first"]["scores"]["v18"] is not None]
            by_label[f"{g}, {label}"] = [round(float(np.mean(scores)), 3) if scores else None, len(scores)]
    words = {g: [len(u["description"].split()) for m in MODELS for u in rows[m] if u["group"] == g and u["description"]]
             for g in GROUPS}
    short = MAX_DESCRIPTION_WORDS + 5
    restricted = {}
    for g in ("real", "pooled 50/50"):
        per_model = [np.array([u["first"]["scores"]["v18"]["pearson"] - u["first"]["scores"]["v15"]["pearson"] for u in rows[m]
                               if u["group"] == g and u["description"] and len(u["description"].split()) <= short
                               and all(u["first"]["scores"][a] is not None for a in ARMS)]) for m in MODELS]
        point, interval = paired_mean(per_model, rng)
        restricted[g] = {"mean over sizes": round(point, 3), "interval": [round(x, 3) for x in interval],
                         "units": int(sum(map(len, per_model)))}
    flag_rates = {m: {g: {r: round(float(np.mean([u["flag"][r] for u in rows[m] if u["group"] == g and u["flag"][r] is not None])), 3)
                          for r in READERS} for g in GROUPS} for m in MODELS}
    return {"tracked neurons": tracked, "judge: share naming unrelated patterns": judge, "v18 score by v18 label": by_label,
            "description words": {g: {"mean": round(float(np.mean(w)), 1), "10th to 90th percentile":
                                      [int(x) for x in np.percentile(w, [10, 90])], f"share over {MAX_DESCRIPTION_WORDS}":
                                      round(float(np.mean([x > MAX_DESCRIPTION_WORDS for x in w])), 3)} for g, w in words.items()},
            f"paired score difference, descriptions of at most {short} words": restricted,
            "C-flag rates by size": flag_rates}


def paired_share_difference(units: list[dict], value, rng) -> dict:
    """Mean of value(u) for v18 minus v14 (or v15) over units, with a bootstrap interval resampling units."""
    d = np.array([value(u) for u in units], float)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)]
    return {"difference": round(float(d.mean()), 3), "interval": [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])],
            "units": len(d)}


def before_first_tool_call(record: dict) -> list[dict]:
    messages = record["messages"]
    return messages[:next((i for i, m in enumerate(messages) if "tool" in m), len(messages))]


def after_review(rows: dict, answers: dict, rng) -> dict:
    """Post hoc, added after an independent review of the first write-up. (1) Whether the descriptions'
    score gain differs between pools and real neurons: the v18-minus-v15 change in the pools-minus-real
    gap, paired (rubric_v8.paired_gain, units resampled within size and type). (2) Paired v18-minus-v14
    differences in the A and C shares. (3) Paired v18-minus-v15 differences in the judge's yes. (4) Traces
    of the protocol in the logs: the first tool call, mentions of candidates in the visible text before the
    first tool call or anywhere, and the deepest rank asked for in the agent's own list."""
    scored = {m: [u for u in rows[m] if all(u["first"]["scores"][a] is not None for a in ARMS)] for m in MODELS}
    gaps = {arm: [tuple(np.array([u["first"]["scores"][arm]["pearson"] for u in scored[m] if u["group"] == g])
                        for g in ("pooled 50/50", "real")) for m in MODELS] for arm in ARMS}
    point, interval, _, _ = paired_gain(gaps["v18"], gaps["v15"], rng)
    shares = {}
    for g in ("real", "pooled 50/50"):
        units = [u for m in MODELS for u in rows[m] if u["group"] == g and None not in u["labels"].values()]
        for label in (ONE_JOB, UNRELATED_JOBS):
            shares[f"{g}, {label}"] = paired_share_difference(
                units, lambda u, x=label: (u["labels"]["v18"] == x) - (u["labels"]["v14"] == x), rng)
    judge = {g: paired_share_difference([u for m in MODELS for u in rows[m] if u["group"] == g and None not in u["judge"].values()],
                                        lambda u: int(u["judge"]["v18"]) - int(u["judge"]["v15"]), rng) for g in GROUPS}
    traces = {}
    for reading in ("first", "retest"):
        records = [u[reading]["agent"] for u in answers["units"].values() if reading in u]
        deepest = [max((int(c["arguments"]["start"]) + int(c["arguments"]["count"]) - 1 for c in r["tool_calls"]
                        if c["tool"] == "examples" and c["answer"] != NO_CALLS_LEFT), default=0) for r in records]
        traces[reading] = {
            "readings": len(records),
            "first tool call is activation": sum(bool(r["tool_calls"]) and r["tool_calls"][0]["tool"] == "activation"
                                                 for r in records),
            "visible text before the first tool call mentions candidates": sum(
                any("candidat" in m.get("text", "").lower() for m in before_first_tool_call(r)) for r in records),
            "visible text anywhere, final reply included, mentions candidates": sum(
                any("candidat" in m.get("text", "").lower() for m in r["messages"]) or "candidat" in r["reply"].lower()
                for r in records),
            f"deepest rank asked for in the agent's list at most {PROTOCOL_RANKS[1]}": sum(d <= PROTOCOL_RANKS[1] for d in deepest)}
    return {"pools-minus-real gap, v18 minus v15 (paired)": {"difference": round(point, 3),
                                                             "interval": [round(x, 3) for x in interval]},
            "label shares, v18 minus v14 (paired)": shares, "judge yes, v18 minus v15 (paired)": judge,
            "protocol traces": traces}


def analyze(answers: dict, v15: dict, v14: dict, v8_auc: dict) -> dict:
    rows = unit_rows(answers, v15, v14)
    rng = np.random.default_rng([18, 99])
    stats = {"primary": primary(rows, rng),
             "v8 label aucs, for context (unpaired)": v8_auc,
             "detection": {measure: {"v18": difference(rows, measure, "v18", rng), "v14": difference(rows, measure, "v14", rng),
                                     "v18 over v14": gain(rows, measure, "v18", "v14", rng)}
                           for measure in ("flag", "not_one_job")},
             "related pools minus real, C flag": {r: difference(rows, "flag", r, rng, ("related", "real")) for r in READERS},
             "one job": one_job(rows, rng),
             "label shares": label_shares(rows),
             "simulation score": score_tests(rows, rng),
             "retest": retest(rows, rng),
             "tool use and cost": tools_and_cost(rows, answers),
             "checks": {**held_out_leak(answers, v15),
                        "units without a v18 answer": sum(u["labels"]["v18"] is None for m in MODELS for u in rows[m]),
                        "units without a v18 score": sum(u["first"]["scores"]["v18"] is None for m in MODELS for u in rows[m]),
                        "v14 prompt hashes of the same units": len({v14["first"][u["id"]]["prompt_sha256"]
                                                                    for m in MODELS for u in rows[m]})},
             "exploratory": exploratory(rows, answers, v15, rng),
             "post hoc": post_hoc(rows, answers, rng),
             "post hoc, after an independent review": after_review(rows, answers, rng)}
    stats["verdict"] = stats["primary"]["verdict"]
    return {"summary": stats, "units": rows}


def main() -> None:
    answers = json.loads(ANSWERS_PATH.read_text())
    v15 = json.loads((RESULTS_DIR / "v15_answers.json").read_text())
    v8 = json.loads((RESULTS_DIR / "classification_v8.json").read_text())["summary"]["post_hoc"]["label_auc"]
    out = analyze(answers, v15, v14_extract(), {r: v8[r] for r in V8_READERS})
    (RESULTS_DIR / "classification_v18.json").write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(json.dumps({k: s[k] for k in ("verdict", "primary", "one job", "retest")}, indent=1))
    print(json.dumps({k: v for k, v in s["simulation score"].items()}, indent=1))


if __name__ == "__main__":
    main()
