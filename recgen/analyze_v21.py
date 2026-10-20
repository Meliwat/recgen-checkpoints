"""Apply the pre-registered v21 rules (recgen/rubric_v21.py): Opus under v7's rubric and the essay's
E3, one shot (arm A, six sizes) and as an investigating agent (arm B, three sizes).

Reads results/v21_answers.json (from recgen.reader_v21), v16's answers (results/v16_answers.json),
v7's Haiku answers, v10's Sonnet cache (data/v10/sonnet/, read only), v8's answers (for v8's prompt
hashes) and v14's Opus agent cache (data/v14/agent_opus/ and agent_opus_retest/, read only). A unit's
size and group come from its id, so no recording is loaded. A unit a series never labeled is left out
of that series and, pairwise, out of paired comparisons.
"""

import json
from collections import Counter

import numpy as np

from recgen.analyze_v8 import CATCH_ALL_WORDS
from recgen.analyze_v16 import (
    POOL, baselines, excess, not_one_main, one_main, patterns_of, rank, retest, rounded, rubric_of, share, strata,
    unit_place, unrelated, v10_sonnet,
)
from recgen.analyze_v16 import auc_over_sizes as auc_over
from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.llm_reader import ANSWERS_PATH as V7_ANSWERS_PATH
from recgen.llm_reader_v16 import ANSWERS_PATH as V16_ANSWERS_PATH
from recgen.reader_v21 import ANSWERS_PATH, v8_prompt_hashes
from recgen.rubric_v5 import MODELS
from recgen.rubric_v8 import MAX_TOOL_CALLS, MIN_TOOL_CALLS
from recgen.rubric_v8 import MODELS as AGENT_MODELS
from recgen.rubric_v16 import LABELS, ONE_MAIN_JOB, bootstrap, changes, near_essay, verdict
from recgen.rubric_v21 import ARM_B, ARMS_A, E3, ESSAY_ONE_MAIN_JOB, GROUPS_A, V7_RUBRIC, V14_AGENT, separates

V14_DIR = DATA_DIR / "v14"
OPUS_V7, OPUS_E3 = "opus-v7", "opus-E3"
REFERENCES_A = ("haiku-E3", "sonnet-E3", "haiku-v7", "sonnet-v7")  # from v16, v7 and v10: other days
GROUPS_B = ("real", POOL, "related")
ESSAY_SHARES = {"one main job": 0.90, "two related jobs": 0.07, "not yet understood": "0.02 to 0.04"}
EXAMPLES = 3


def v14_agent(hashes: dict[str, str]) -> dict:
    """v14's Opus agent readings ({"first", "retest"}: unit id -> record), kept only where the
    reading's prompt is v8's for the unit (by hash), with a count of those left out."""
    out = {"first": {}, "retest": {}, "left out": 0}
    for part, folder in (("first", "agent_opus"), ("retest", "agent_opus_retest")):
        for path in sorted((V14_DIR / folder).glob("*.json")):
            record = json.loads(path.read_text())
            if record["prompt_sha256"] == hashes.get(record["id"]):
                out[part][record["id"]] = record
            else:
                out["left out"] += 1
    return out


def rows_a(answers: dict, v16: dict, base: dict) -> list[dict]:
    """Arm A's 900 units (v7's real neurons and 50/50 pools) with each series' first-reading label."""
    rows = []
    for unit_id in base["haiku-v7"]["first"]:
        size, group = unit_place(unit_id)
        if group not in GROUPS_A:
            continue
        labels = {s: (answers["arm_a"][s]["first"].get(unit_id) or {}).get("label") for s in ARMS_A}
        labels |= {s: (v16["arms"][s]["first"].get(unit_id) or {}).get("label") for s in REFERENCES_A if s in v16["arms"]}
        labels |= {s: b["first"].get(unit_id) for s, b in base.items()}
        rows.append({"id": unit_id, "size": size, "group": group, "labels": labels})
    return rows


def rows_b(answers: dict, v14: dict, unit_ids: list[str]) -> list[dict]:
    """Arm B's 300 units (v8's) with the Opus agent's E3 label, v14's A/B/C label, and one-shot Opus's
    labels from arm A where arm A read the unit (real neurons and 50/50 pools)."""
    rows = []
    for unit_id in unit_ids:
        size, group = unit_place(unit_id)
        labels = {ARM_B: (answers["arm_b"]["first"].get(unit_id) or {}).get("label"),
                  V14_AGENT: (v14["first"].get(unit_id) or {}).get("label")}
        labels |= {s: (answers["arm_a"][s]["first"].get(unit_id) or {}).get("label") for s in ARMS_A}
        rows.append({"id": unit_id, "size": size, "group": group, "labels": labels})
    return rows


def series_stats(rows: list[dict], series: str, sizes, groups, rng) -> dict:
    """Tests A1, A3, B1, B2, S1, S2 and S8 for one series: one-main-job shares by group (over sizes and
    per size), label shares, detection, the label AUC and its verdicts."""
    out = {"one_main_job": {}, "label_shares": {}, "labels": {}, "detection": {}}
    for group in groups:
        out["labels"][group] = dict(sorted(Counter(u["labels"][series] for u in rows if u["group"] == group).items(),
                                           key=lambda kv: str(kv[0])))
        st = strata(rows, [series], [group], one_main, sizes)
        if not all(len(v) for v in st.values()):
            out["one_main_job"][group] = {"value": None, "units": int(sum(len(v) for v in st.values()))}
            continue
        out["one_main_job"][group] = {
            **bootstrap(st, lambda s, g=group: share(s, g), rng),
            "per_size": {size: bootstrap({k: v for k, v in st.items() if k[0] == size}, lambda s, g=group: share(s, g), rng)
                         for size in sizes},
            "units": int(sum(len(v) for v in st.values()))}
        if group in GROUPS_A:
            out["label_shares"][group] = {
                label: bootstrap(strata(rows, [series], [group], lambda _, x, lab=label: float(x == lab), sizes),
                                 lambda s, g=group: share(s, g), rng) for label in LABELS[rubric_of(series)]}
    out["near_essay"] = near_essay(out["one_main_job"]["real"]["interval"])
    flags = {"not one main job": not_one_main} | ({"unrelated jobs (C)": unrelated} if rubric_of(series) == V7_RUBRIC else {})
    for name, value in flags.items():
        st = strata(rows, [series], [POOL, "real"], value, sizes)
        out["detection"][name] = bootstrap(st, excess, rng)
    out["label_auc"] = bootstrap(strata(rows, [series], [POOL, "real"], rank, sizes), auc_over, rng)
    out["separates"] = separates(out["label_auc"]["interval"])
    return out


def versus(rows: list[dict], a: str, b: str, sizes, rng) -> dict:
    """Paired, on units both series labeled: a minus b in the one-main-job share of real neurons
    (rise_real) and of 50/50 pools (rise_pool), their difference, and the label AUC difference."""
    st = strata(rows, [a, b], ["real", POOL], one_main, sizes)

    def rise(s, group):
        return share(s, group, 0) - share(s, group, 1)

    ranks = strata(rows, [a, b], [POOL, "real"], rank, sizes)
    auc_difference = bootstrap(ranks, lambda s: auc_over(s, 0) - auc_over(s, 1), rng)
    return {"rise_real": bootstrap(st, lambda s: rise(s, "real"), rng),
            "rise_pool": bootstrap(st, lambda s: rise(s, POOL), rng),
            "rise_pool_minus_rise_real": bootstrap(st, lambda s: rise(s, POOL) - rise(s, "real"), rng),
            "label_auc_difference": {**auc_difference, "changes": changes(auc_difference["interval"])},
            "units": int(sum(len(v) for v in st.values()))}


def primary(rows: list[dict], a: str, b: str, sizes, rng) -> dict:
    """A2 and A3 (or B3): the paired comparison with v16's verdict rule."""
    out = versus(rows, a, b, sizes, rng)
    out["verdict"] = verdict(out["rise_real"]["interval"], out["rise_pool_minus_rise_real"]["interval"])
    return out


def not_one(label: str) -> bool:
    return label != ONE_MAIN_JOB


def retests(answers: dict, v14: dict, rng) -> dict:
    """S3: agreement of two readings, with the "not one main job" flag."""
    out = {}
    for s in ARMS_A:
        parts = answers["arm_a"][s]
        out[s] = retest({k: a["label"] for k, a in parts["first"].items()},
                        {k: a["label"] for k, a in parts["retest"].items()}, not_one, rng)
    for s, parts in ((ARM_B, answers["arm_b"]), (V14_AGENT, v14)):
        out[s] = retest({k: r["label"] for k, r in parts["first"].items()},
                        {k: r["label"] for k, r in parts["retest"].items()}, not_one, rng)
    return out


def tool_use(records: list[dict]) -> dict:
    """S4: v8's tool-use measures (recgen/analyze_v8), plus the share using all of the budget."""
    calls = [r["tool_calls"] for r in records]
    if not calls:
        return {"readings": 0}
    return {
        "readings": len(calls),
        "mean tool calls": float(np.mean([len(c) for c in calls])),
        f"share with at least {MIN_TOOL_CALLS}": float(np.mean([len(c) >= MIN_TOOL_CALLS for c in calls])),
        f"share using all {MAX_TOOL_CALLS}": float(np.mean([len(c) >= MAX_TOOL_CALLS for c in calls])),
        "share writing test text": float(np.mean([any(x["tool"] == "activation" for x in c) for c in calls])),
        "share reading past rank 50": float(np.mean([any(x["tool"] == "examples" and int(x["arguments"].get("start", 1))
                                                         + int(x["arguments"].get("count", 1)) > 51 for x in c) for c in calls])),
        "mean test texts": float(np.mean([sum(len(x["arguments"].get("texts") or []) for x in c if x["tool"] == "activation")
                                          for c in calls]))}


def tool_use_by_group(first: dict, unit_ids: list[str]) -> dict:
    out = {"all": tool_use([first[k] for k in unit_ids if k in first and first[k]["label"] is not None])}
    for group in GROUPS_B:
        out[group] = tool_use([first[k] for k in unit_ids if k in first and first[k]["label"] is not None
                               and unit_place(k)[1] == group])
    return out


def usage(answers: dict, v14: dict) -> dict:
    """S5: CLI-reported cost and tokens per series, the pilots, and the spend log's total."""
    out = {}
    for s, parts in answers["arm_a"].items():
        done = list(parts["first"].values()) + list(parts["retest"].values())
        out[s] = {"first readings": len(parts["first"]), "retests": len(parts["retest"]),
                  "without a label": sum(a["label"] is None for a in done),
                  "models": sorted({m for a in done for m in a["models"]}),
                  "cost_usd": sum(a["cost_usd"] or 0 for a in done),
                  "input_tokens": sum(a["input_tokens"] for a in done),
                  "mean output tokens": float(np.mean([a["output_tokens"] for a in done])) if done else None,
                  "mean thinking tokens": float(np.mean([a["thinking_tokens"] for a in done])) if done else None}
    for s, parts in ((ARM_B, answers["arm_b"]), (V14_AGENT, v14)):
        done = [r for part in ("first", "retest") for r in parts[part].values()]
        out[s] = {"first readings": len(parts["first"]), "retests": len(parts["retest"]),
                  "without a label": sum(r["label"] is None for r in done),
                  "stopped by the spend guard": sum(r["error"] == "spend cap reached" for r in done),
                  "with retries": sum(len(r["attempts"]) > 1 for r in done),
                  "all offered both tools": all(len(r["tools_offered"]) == 2 for r in done if r["label"]),
                  "models": sorted({m for r in done for m in r["models"]}),
                  "cost_usd": sum(r["cost_usd"] for r in done),
                  "input_tokens": sum(r["input_tokens"] for r in done),
                  "mean output tokens": float(np.mean([r["output_tokens"] for r in done])) if done else None}
    pilot_a = [a for part in answers["pilot"]["arm_a"].values() for a in part.values()]
    pilot_b = list(answers["pilot"]["arm_b"].values())
    out["pilots"] = {"arm A readings": len(pilot_a), "arm A labeled": sum(a["label"] is not None for a in pilot_a),
                     "arm A cost_usd": sum(a["cost_usd"] or 0 for a in pilot_a),
                     "arm B readings": len(pilot_b), "arm B labeled": sum(r["label"] is not None for r in pilot_b),
                     "arm B cost_usd": sum(r["cost_usd"] for r in pilot_b)}
    out["all v21 CLI calls, by the spend log and arm B's readings"] = answers["spend_usd"]
    return out


def prompt_checks(answers: dict, v7_answers: dict, v16: dict, v14: dict) -> dict:
    """Arm A's prompts against v7's (v7 rubric) and v16's (E3) for the same units, by hash; arm B's
    prompts were checked against v8's by the runner before reading."""
    def matches(series, reference):
        first = answers["arm_a"][series]["first"]
        return {"units": len(first), "same prompt": sum(a["prompt_sha256"] == reference[k]["prompt_sha256"]
                                                        for k, a in first.items() if k in reference)}
    return {OPUS_V7: matches(OPUS_V7, v7_answers["first"]), OPUS_E3: matches(OPUS_E3, v16["arms"]["haiku-E3"]["first"]),
            "v14 readings left out (prompt not v8's)": v14["left out"]}


def one_main_job_patterns(rows: list[dict], series: str, patterns: dict) -> dict:
    """Exploratory: per group, how many patterns one-main-job answers list, and v8's crude catch-all
    keyword share among them."""
    out = {}
    for group in GROUPS_A:
        ids = [u["id"] for u in rows if u["group"] == group and u["labels"][series] == ONE_MAIN_JOB and u["id"] in patterns]
        lists = [patterns[k] for k in ids]
        out[group] = {"answers": len(ids),
                      "mean patterns": float(np.mean([len(p) for p in lists])) if lists else None,
                      "share listing 2 or more": float(np.mean([len(p) >= 2 for p in lists])) if lists else None,
                      "share with a catch-all word": float(np.mean([any(w in " ".join(p).lower() for w in CATCH_ALL_WORDS)
                                                                    for p in lists])) if lists else None}
    return out


def exploratory(ra: list[dict], rb: list[dict], answers: dict, v14: dict) -> dict:
    """Listed in the rubric: label moves from v7's rubric to E3, patterns of one-main-job answers,
    the catch-all count, arm B's labels by confidence, and examples of pools called one main job."""
    patterns = {s: {k: patterns_of(a["reply"]) for k, a in answers["arm_a"][s]["first"].items()} for s in ARMS_A}
    patterns[ARM_B] = {k: r["patterns"] for k, r in answers["arm_b"]["first"].items()}
    patterns[V14_AGENT] = {k: r["patterns"] for k, r in v14["first"].items()}
    out = {"label_moves": {}, "one_main_job_answers": {}, "labels_by_confidence": {}, "examples": {}}
    for name, rows, a, b, groups in (("arm A: v7 rubric -> E3", ra, OPUS_V7, OPUS_E3, GROUPS_A),
                                     ("arm B: v14 (v7 rubric) -> E3", rb, V14_AGENT, ARM_B, GROUPS_B)):
        out["label_moves"][name] = {g: dict(sorted(Counter(
            f"{u['labels'][a]} -> {u['labels'][b]}" for u in rows if u["group"] == g
            and u["labels"][a] is not None and u["labels"][b] is not None).items())) for g in groups}
    for s, rows in ((OPUS_V7, ra), (OPUS_E3, ra), (ARM_B, rb), (V14_AGENT, rb)):
        out["one_main_job_answers"][s] = one_main_job_patterns(rows, s, patterns[s])
        pools = [u for u in rows if u["group"] == POOL and u["labels"][s] == ONE_MAIN_JOB]
        out["examples"][s] = [{"id": u["id"], "patterns": patterns[s][u["id"]]}
                              for u in pools[:: max(1, len(pools) // EXAMPLES)][:EXAMPLES]]
    for s, first in ((ARM_B, answers["arm_b"]["first"]), (V14_AGENT, v14["first"])):
        out["labels_by_confidence"][s] = {
            g: {level: dict(Counter(r["label"] for k, r in first.items() if unit_place(k)[1] == g
                                    and r["confidence"] == level and r["label"])) for level in ("low", "medium", "high")}
            for g in GROUPS_A}
    return out


def analyze(answers: dict, v7_answers: dict, v16: dict, sonnet: dict | None, v14: dict, unit_ids_b: list[str], rng) -> dict:
    base = baselines(v7_answers, sonnet)
    ra, rb = rows_a(answers, v16, base), rows_b(answers, v14, unit_ids_b)
    series_a = list(ARMS_A) + [s for s in REFERENCES_A if any(u["labels"].get(s) is not None for u in ra)]
    summary = {
        "arm_a": {s: series_stats(ra, s, MODELS, GROUPS_A, rng) for s in series_a},
        "arm_b": {s: series_stats(rb, s, AGENT_MODELS, GROUPS_B, rng) for s in (ARM_B, V14_AGENT)},
        "primary_a": primary(ra, OPUS_E3, OPUS_V7, MODELS, rng),
        "primary_b": primary(rb, ARM_B, V14_AGENT, AGENT_MODELS, rng),
        "readers_same_units": {f"{a} minus {b}": versus(ra, a, b, MODELS, rng)
                               for a, b in ((OPUS_E3, "sonnet-E3"), (OPUS_E3, "haiku-E3"), (OPUS_V7, "sonnet-v7"),
                                            (OPUS_V7, "haiku-v7")) if b in series_a},
        "investigation_same_units": {
            f"{a} minus {b}": {**versus(rb, a, b, AGENT_MODELS, rng),
                               "one shot": series_stats([u for u in rb if u["group"] in GROUPS_A], b, AGENT_MODELS,
                                                        GROUPS_A, rng)["one_main_job"]}
            for a, b in ((ARM_B, OPUS_E3), (V14_AGENT, OPUS_V7))},
        "retest": retests(answers, v14, rng),
        "essay_one_main_job": ESSAY_ONE_MAIN_JOB, "essay_shares": ESSAY_SHARES,
        "sonnet_v7_rubric_baseline": "v10's first readings" if sonnet else "not available (v10's cache incomplete)",
    }
    first_b = answers["arm_b"]["first"]
    return {"summary": rounded(summary),
            "tool_use": rounded({ARM_B: tool_use_by_group(first_b, unit_ids_b),
                                 V14_AGENT: tool_use_by_group(v14["first"], unit_ids_b)}),
            "exploratory": rounded(exploratory(ra, rb, answers, v14)),
            "usage": rounded(usage(answers, v14)),
            "checks": prompt_checks(answers, v7_answers, v16, v14),
            "units": {"arm_a": ra, "arm_b": rb}}


def main() -> None:
    answers = json.loads(ANSWERS_PATH.read_text())
    v7_answers = json.loads(V7_ANSWERS_PATH.read_text())
    v16 = json.loads(V16_ANSWERS_PATH.read_text())
    hashes = v8_prompt_hashes()
    v8_ids = list(json.loads((RESULTS_DIR / "v8_agent_answers.json").read_text())["sonnet"]["first"])
    result = analyze(answers, v7_answers, v16, v10_sonnet(v7_answers), v14_agent(hashes), v8_ids,
                     np.random.default_rng([21, 1]))
    (RESULTS_DIR / "classification_v21.json").write_text(json.dumps(result, indent=1))
    s = result["summary"]
    for arm in ("arm_a", "arm_b"):
        for name, st in s[arm].items():
            real, pool = st["one_main_job"]["real"], st["one_main_job"][POOL]
            print(f"{name}: real one main job {real['value']:.3f} {real['interval']}, 50/50 {pool['value']:.3f} "
                  f"{pool['interval']}, AUC {st['label_auc']['value']:.3f} {st['label_auc']['interval']}")
    for name in ("primary_a", "primary_b"):
        v = s[name]
        print(f"{name}: {v['verdict']}: rise real {v['rise_real']['value']:+.3f} {v['rise_real']['interval']}, rise pool "
              f"{v['rise_pool']['value']:+.3f} {v['rise_pool']['interval']}, AUC difference "
              f"{v['label_auc_difference']['value']:+.3f} {v['label_auc_difference']['interval']}")
    print(json.dumps(result["usage"], indent=1))
    print(json.dumps(result["checks"], indent=1))


if __name__ == "__main__":
    main()
