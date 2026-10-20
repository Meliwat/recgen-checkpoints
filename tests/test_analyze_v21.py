import json

import numpy as np

from recgen import analyze_v21
from recgen.analyze_v16 import POOL, baselines
from recgen.analyze_v21 import (
    OPUS_E3, OPUS_V7, analyze, primary, retests, rows_a, rows_b, series_stats, tool_use, v14_agent, versus,
)
from recgen.rubric_v5 import MODELS
from recgen.rubric_v8 import MODELS as AGENT_MODELS
from recgen.rubric_v16 import INFLATES, NO_RISE, SELECTIVE
from recgen.rubric_v21 import ARM_B, ARMS_A, V14_AGENT

KINDS = {"real": "real::", POOL: "pooled:50/50:", "related": "related:50/50:", "pooled 90/10": "pooled:90/10:"}


def unit_ids(sizes, groups, per_group: int = 4) -> list[str]:
    return [f"{size}:{KINDS[g]}{i}-{i + 1}" for size in sizes for g in groups for i in range(per_group)]


def reply(label) -> dict:
    return {"label": label, "reply": json.dumps({"label": label, "patterns": ["x", "y"]}), "models": ["m"],
            "prompt_sha256": "h", "cost_usd": 0.01, "input_tokens": 10, "output_tokens": 5, "thinking_tokens": 2}


def record(unit_id: str, label, calls: int = 4) -> dict:
    tool_calls = [{"tool": "examples", "arguments": {"start": 51, "count": 10}, "answer": "x"}]
    tool_calls += [{"tool": "activation", "arguments": {"texts": ["a", "b"]}, "answer": "x"}] * (calls - 1)
    return {"id": unit_id, "label": label, "patterns": ["x"], "confidence": "high", "error": None, "attempts": [{}],
            "cost_usd": 0.1, "input_tokens": 100, "output_tokens": 50, "models": ["m"], "tools_offered": ["a", "b"],
            "prompt_sha256": f"v8{unit_id}", "tool_calls": tool_calls}


def fake(a_labels: dict, b_labels: dict, v14_labels: dict, v7: str = "C") -> tuple:
    """answers, v7 answers, v16 answers, v14 readings and v8's unit ids, where every unit of a group
    gets one label per series. a_labels: arm A series -> group -> label; b_labels and v14_labels:
    group -> label."""
    ids_a = unit_ids(MODELS, ("real", POOL, "pooled 90/10"))
    ids_b = unit_ids(AGENT_MODELS, ("real", POOL, "related"))
    group_of = {k: analyze_v21.unit_place(k)[1] for k in ids_a + ids_b}
    answers = {"arm_a": {s: {"first": {}, "retest": {}} for s in ARMS_A}, "arm_b": {"first": {}, "retest": {}},
               "pilot": {"arm_a": {s: {} for s in ARMS_A}, "arm_b": {}}, "spend_usd": 1.0}
    for s, by_group in a_labels.items():
        for k in ids_a + ids_b:
            if group_of[k] in by_group:
                answers["arm_a"][s]["first"][k] = reply(by_group[group_of[k]])
    answers["arm_b"]["first"] = {k: record(k, b_labels[group_of[k]]) for k in ids_b}
    answers["arm_b"]["retest"] = {k: record(k, b_labels[group_of[k]]) for k in ids_b[:6]}
    v14 = {"first": {k: record(k, v14_labels[group_of[k]]) for k in ids_b},
           "retest": {k: record(k, v14_labels[group_of[k]]) for k in ids_b[:6]}, "left out": 0}
    v7_answers = {"first": {k: {"label": v7, "prompt_sha256": "h"} for k in set(ids_a + ids_b)}, "retest": {}}
    v16 = {"arms": {"haiku-E3": {"first": {k: reply("C") for k in ids_a}, "retest": {}}}}
    return answers, v7_answers, v16, v14, ids_b


def test_rows_join_series_on_unit_ids_and_keep_missing_labels_missing():
    answers, v7_answers, v16, v14, ids_b = fake({OPUS_E3: {"real": "A", POOL: "C"}}, {"real": "A", POOL: "B", "related": "A"},
                                                {"real": "B", POOL: "C", "related": "A"})
    ra = rows_a(answers, v16, baselines(v7_answers, None))
    assert len(ra) == len(MODELS) * 8 and {u["group"] for u in ra} == {"real", POOL}
    assert all(u["labels"][OPUS_E3] == "A" and u["labels"][OPUS_V7] is None and u["labels"]["haiku-v7"] == "C"
               for u in ra if u["group"] == "real")
    rb = rows_b(answers, v14, ids_b)
    assert len(rb) == 36 and all(u["labels"][ARM_B] == "B" and u["labels"][V14_AGENT] == "C" for u in rb if u["group"] == POOL)
    assert all(u["labels"][OPUS_E3] == "A" for u in rb if u["group"] == "real")
    assert all(u["labels"][OPUS_E3] is None for u in rb if u["group"] == "related")


def test_series_stats_shares_detection_auc_and_verdicts():
    answers, v7_answers, v16, v14, ids_b = fake({OPUS_E3: {"real": "A", POOL: "C"}}, {"real": "A", POOL: "C", "related": "A"},
                                                {"real": "A", POOL: "C", "related": "A"})
    ra = rows_a(answers, v16, baselines(v7_answers, None))
    stats = series_stats(ra, OPUS_E3, MODELS, ("real", POOL), np.random.default_rng(0))
    assert stats["one_main_job"]["real"]["value"] == 1.0 and stats["one_main_job"][POOL]["value"] == 0.0
    assert stats["near_essay"] and stats["separates"] and stats["label_auc"]["value"] == 1.0
    assert stats["label_shares"][POOL]["C"]["value"] == 1.0 and stats["label_shares"]["real"]["B"]["value"] == 0.0
    assert stats["detection"]["not one main job"]["value"] == 1.0 and "unrelated jobs (C)" not in stats["detection"]
    rb = rows_b(answers, v14, ids_b)
    b = series_stats(rb, V14_AGENT, AGENT_MODELS, ("real", POOL, "related"), np.random.default_rng(0))
    assert b["detection"]["unrelated jobs (C)"]["value"] == 1.0 and set(b["one_main_job"]["real"]["per_size"]) == set(AGENT_MODELS)
    assert b["one_main_job"]["related"]["value"] == 1.0


def test_primary_verdicts_follow_v16s_rule():
    rng = np.random.default_rng(0)
    same = fake({OPUS_E3: {"real": "A", POOL: "A"}, OPUS_V7: {"real": "C", POOL: "C"}}, {"real": "A", POOL: "C", "related": "A"},
                {"real": "A", POOL: "C", "related": "A"})
    ra = rows_a(same[0], same[2], baselines(same[1], None))
    out = primary(ra, OPUS_E3, OPUS_V7, MODELS, rng)
    assert out["rise_real"]["value"] == 1.0 and out["rise_pool"]["value"] == 1.0 and out["verdict"] == INFLATES
    assert out["label_auc_difference"]["value"] == 0.0 and out["label_auc_difference"]["changes"] is False
    rb = rows_b(same[0], same[3], same[4])
    assert primary(rb, ARM_B, V14_AGENT, AGENT_MODELS, rng)["verdict"] == NO_RISE
    apart = fake({OPUS_E3: {"real": "A", POOL: "C"}, OPUS_V7: {"real": "C", POOL: "C"}}, {"real": "A", POOL: "C", "related": "A"},
                 {"real": "A", POOL: "C", "related": "A"})
    ra = rows_a(apart[0], apart[2], baselines(apart[1], None))
    out = primary(ra, OPUS_E3, OPUS_V7, MODELS, rng)
    assert out["verdict"] == SELECTIVE and out["label_auc_difference"]["value"] == 0.5


def test_versus_is_paired_on_units_both_series_labeled():
    answers, v7_answers, v16, v14, ids_b = fake({OPUS_E3: {"real": "A", POOL: "C"}}, {"real": "A", POOL: "C", "related": "A"},
                                                {"real": "A", POOL: "C", "related": "A"})
    rb = rows_b(answers, v14, ids_b)
    rb[0]["labels"][ARM_B] = None
    out = versus(rb, ARM_B, OPUS_E3, AGENT_MODELS, np.random.default_rng(0))
    assert out["units"] == 23 and out["rise_real"]["value"] == 0.0


def test_v14_readings_need_v8s_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_v21, "V14_DIR", tmp_path)
    for folder in ("agent_opus", "agent_opus_retest"):
        (tmp_path / folder).mkdir()
    for unit_id, digest in (("70m:real::1", "a"), ("70m:real::2", "other")):
        (tmp_path / "agent_opus" / f"{unit_id.replace(':', '_')}.json").write_text(
            json.dumps({"id": unit_id, "prompt_sha256": digest, "label": "A"}))
    out = v14_agent({"70m:real::1": "a", "70m:real::2": "b"})
    assert list(out["first"]) == ["70m:real::1"] and out["left out"] == 1 and out["retest"] == {}


def test_tool_use_measures():
    out = tool_use([record("70m:real::1", "A", calls=4), record("70m:real::2", "A", calls=8)])
    assert out["mean tool calls"] == 6.0 and out["share with at least 4"] == 1.0 and out["share using all 8"] == 0.5
    assert out["share writing test text"] == 1.0 and out["share reading past rank 50"] == 1.0
    assert out["mean test texts"] == 2 * (3 + 7) / 2


def test_retests_use_the_not_one_main_job_flag():
    answers, _, _, v14, _ = fake({OPUS_E3: {"real": "A", POOL: "C"}}, {"real": "A", POOL: "C", "related": "B"},
                                 {"real": "A", POOL: "C", "related": "A"})
    answers["arm_a"][OPUS_E3]["retest"] = {k: reply("B") for k in list(answers["arm_a"][OPUS_E3]["first"])[:4]}
    out = retests(answers, v14, np.random.default_rng(0))
    assert out[ARM_B]["units"] == 6 and out[ARM_B]["same_label"] == 1.0
    assert out[OPUS_E3]["units"] == 4 and out[OPUS_E3]["same_flag"] == 0.0  # real A then B
    assert out[V14_AGENT]["units"] == 6 and out[V14_AGENT]["flag_kappa"] == 1.0


def test_analyze_runs_end_to_end():
    answers, v7_answers, v16, v14, ids_b = fake(
        {OPUS_E3: {"real": "A", POOL: "C"}, OPUS_V7: {"real": "B", POOL: "C"}}, {"real": "A", POOL: "C", "related": "A"},
        {"real": "B", POOL: "C", "related": "A"})
    result = analyze(answers, v7_answers, v16, None, v14, ids_b, np.random.default_rng(0))
    s = result["summary"]
    assert set(s["arm_a"]) == {OPUS_V7, OPUS_E3, "haiku-E3", "haiku-v7"} and s["primary_a"]["verdict"] == SELECTIVE
    assert s["primary_b"]["rise_real"]["value"] == 1.0 and "opus-E3 minus haiku-E3" in s["readers_same_units"]
    assert s["investigation_same_units"][f"{ARM_B} minus {OPUS_E3}"]["rise_real"]["value"] == 0.0
    assert result["checks"][OPUS_E3] == {"units": len(answers["arm_a"][OPUS_E3]["first"]), "same prompt": 24 * 2}
    assert result["exploratory"]["label_moves"]["arm A: v7 rubric -> E3"]["real"] == {"B -> A": 24}
    assert result["usage"][ARM_B]["cost_usd"] == 4.2 and result["tool_use"][ARM_B]["all"]["readings"] == 36
