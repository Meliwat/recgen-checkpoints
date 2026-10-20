import json

import numpy as np

from recgen import analyze_v16
from recgen.analyze_v16 import (
    POOL, analyze, arm_stats, auc_differences, baselines, concurrent_addendum, e4_versus_e3, odds_ratios, patterns_of,
    retest, swapped_auc, thinking_split, unit_place, unit_rows, v7_haiku_thinking, v10_sonnet, versus_v7,
)
from recgen.llm_reader import cache_path
from recgen.rubric_v5 import MODELS
from recgen.rubric_v16 import ARMS, INFLATES, NO_RISE, SELECTIVE
from recgen.rubric_v16_addendum import ARMS as ADDENDUM_ARMS
from recgen.rubric_v16_addendum import NOT_REPRODUCED, REPRODUCED

KINDS = {"real": "real::", POOL: "pooled:50/50:", "pooled 90/10": "pooled:90/10:"}


def fake(labels: dict, per_group: int = 4, v7: dict | None = None) -> tuple[dict, dict]:
    """answers and v7 answers where every unit of a group gets the same label in an arm.
    labels: arm -> group -> label; v7: group -> v7-rubric label."""
    v7 = v7 or {"real": "C", POOL: "C", "pooled 90/10": "C"}
    answers = {"arms": {arm: {"first": {}, "retest": {}} for arm in ARMS}, "pilot": {arm: {} for arm in ARMS},
               "spend_usd": 0.0}
    v7_answers = {"first": {}, "retest": {}}
    for size in MODELS:
        for group, kind in KINDS.items():
            for i in range(per_group):
                unit_id = f"{size}:{kind}{i}-{i + 1}"
                v7_answers["first"][unit_id] = {"label": v7[group], "prompt_sha256": f"h{unit_id}"}
                for arm, by_group in labels.items():
                    label = by_group.get(group)
                    answers["arms"][arm]["first"][unit_id] = {
                        "label": label, "reply": json.dumps({"label": label, "patterns": ["x", "y"]}),
                        "models": ["m"], "cost_usd": 0.001, "input_tokens": 10, "output_tokens": 5, "thinking_tokens": 2}
    return answers, v7_answers


def test_unit_place_reads_size_and_group_from_the_id():
    assert unit_place("2.8b:real::17") == ("2.8b", "real")
    assert unit_place("70m:pooled:70/30:1-2") == ("70m", "pooled 70/30")
    assert unit_place("1b:related:50/50:3-4") == ("1b", "related")


def test_rows_keep_missing_labels_missing():
    answers, v7_answers = fake({"haiku-E3": {"real": "A", POOL: "C"}})
    rows = unit_rows(answers, baselines(v7_answers, None))
    real = [r for r in rows if r["group"] == "real"]
    assert len(rows) == len(MODELS) * 12 and all(r["labels"]["haiku-E3"] == "A" for r in real)
    assert all(r["labels"]["haiku-E4"] is None and r["labels"]["haiku-v7"] == "C" for r in real)
    assert all(r["labels"]["haiku-E3"] is None for r in rows if r["group"] == "pooled 90/10")


def test_arm_stats_shares_detection_and_auc():
    answers, v7_answers = fake({"haiku-E4": {"real": "A", POOL: "D", "pooled 90/10": "A"}})
    stats = arm_stats(unit_rows(answers, baselines(v7_answers, None)), "haiku-E4", np.random.default_rng(0))
    assert stats["one_main_job"]["real"]["value"] == 1.0 and stats["one_main_job"][POOL]["value"] == 0.0
    assert stats["one_main_job"]["real"]["per_size"]["70m"]["interval"] == [1.0, 1.0]
    assert stats["near_essay"] and stats["labels"]["real"] == {"A": 4 * len(MODELS)}
    assert stats["detection"]["not one main job"]["value"] == 1.0 and stats["detection"]["unrelated jobs"]["value"] == 1.0
    assert stats["label_auc"]["value"] == 1.0


def test_verdict_inflates_when_pools_rise_as_much_and_selective_when_they_dont():
    rng = np.random.default_rng(0)
    answers, v7_answers = fake({"haiku-E3": {"real": "A", POOL: "A"}})
    out = versus_v7(unit_rows(answers, baselines(v7_answers, None)), "haiku", "E3", rng)
    assert out["rise_real"]["value"] == 1.0 and out["rise_pool_minus_rise_real"]["interval"] == [0.0, 0.0]
    assert out["verdict"] == INFLATES
    answers, v7_answers = fake({"haiku-E3": {"real": "A", POOL: "C"}})
    out = versus_v7(unit_rows(answers, baselines(v7_answers, None)), "haiku", "E3", rng)
    assert out["rise_pool"]["value"] == 0.0 and out["verdict"] == SELECTIVE


def test_e4_versus_e3_is_paired_and_compares_unrelated_flags_with_v7():
    answers, v7_answers = fake({"haiku-E3": {"real": "A", POOL: "A"}, "haiku-E4": {"real": "A", POOL: "D"}},
                               v7={"real": "A", POOL: "C", "pooled 90/10": "A"})
    out = e4_versus_e3(unit_rows(answers, baselines(v7_answers, None)), "haiku", np.random.default_rng(0))
    assert out["real one main job"]["value"] == 0.0 and not out["real one main job"]["changes"]
    assert out["50/50 one main job"]["value"] == -1.0 and out["50/50 one main job"]["changes"]
    assert out["detection, not one main job"]["value"] == 1.0
    assert out["detection, unrelated jobs, E4 minus v7 rubric"]["value"] == 0.0  # D on pools as v7's C on pools


def test_retest_agreement():
    first = {"70m:real::1": "A", "70m:real::2": "C", "70m:pooled:50/50:1-2": "D", "70m:real::3": None}
    second = {"70m:real::1": "A", "70m:real::2": "B", "70m:pooled:50/50:1-2": "D", "70m:real::3": "A"}
    out = retest(first, second, lambda x: x != "A", np.random.default_rng(0))
    assert out["units"] == 3 and out["real units"] == 2 and out["same_flag"] == 1.0 and out["flag_kappa"] == 1.0
    assert abs(out["same_label"] - 2 / 3) < 1e-9


def test_v10_baseline_needs_every_unit_with_v7s_exact_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_v16, "V10_SONNET_DIR", tmp_path)
    v7_answers = {"first": {"70m:real::1": {"prompt_sha256": "a"}, "70m:real::2": {"prompt_sha256": "b"}},
                  "retest": {"70m:real::1": {"prompt_sha256": "a"}}}
    (tmp_path / "llm").mkdir()
    (tmp_path / "llm_retest").mkdir()
    for unit_id, digest in (("70m:real::1", "a"), ("70m:real::2", "b")):
        cache_path(tmp_path / "llm", unit_id).write_text(json.dumps({"prompt_sha256": digest, "label": "B", "models": ["s"]}))
    out = v10_sonnet(v7_answers)
    assert out["first"] == {"70m:real::1": "B", "70m:real::2": "B"} and out["retest"] == {} and out["models"] == ["s"]
    cache_path(tmp_path / "llm", "70m:real::2").write_text(json.dumps({"prompt_sha256": "other", "label": "B", "models": []}))
    assert v10_sonnet(v7_answers) is None


def test_patterns_of_reads_the_reply_json():
    assert patterns_of('Sure:\n{"label": "A", "patterns": ["a", "b"]}') == ["a", "b"]
    assert patterns_of("no json") == [] and patterns_of('{"label": "A"}') == []


def test_analyze_runs_end_to_end_without_a_sonnet_baseline():
    labels = {arm: {"real": "A", POOL: "B", "pooled 90/10": "A"} for arm in ARMS}
    answers, v7_answers = fake(labels)
    result = analyze(answers, v7_answers, None, np.random.default_rng(0))
    assert result["summary"]["versus_v7"]["sonnet-E3"].startswith("not run")
    assert result["summary"]["versus_v7"]["haiku-E3"]["verdict"] == SELECTIVE
    assert result["usage"]["haiku-E3"]["first readings"] == len(MODELS) * 12
    assert result["concurrent_addendum"].startswith("not run") and "thinking" in result["post_hoc"]


def test_auc_differences_are_paired_per_reader():
    answers, v7_answers = fake({"haiku-E3": {"real": "A", POOL: "C"}, "haiku-E4": {"real": "A", POOL: "A"}},
                               v7={"real": "A", POOL: "C", "pooled 90/10": "C"})
    base = baselines(v7_answers, None)
    out = auc_differences(unit_rows(answers, base), base, np.random.default_rng(0))
    assert out["haiku: E3 minus E4"]["value"] == 0.5 and out["haiku: E3 minus v7"]["interval"] == [0.0, 0.0]
    assert out["haiku: E4 minus v7"]["value"] == -0.5 and not any(k.startswith("sonnet") for k in out)


def test_v7_haiku_thinking_reads_v7s_cache_for_v7s_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze_v16, "V7_LLM_DIR", tmp_path)
    v7_answers = {"first": {"70m:real::1": {"prompt_sha256": "a"}, "70m:real::2": {"prompt_sha256": "b"},
                            "70m:real::3": {"prompt_sha256": "c"}}}
    for unit_id, digest in (("70m:real::1", "a"), ("70m:real::2", "other")):
        cache_path(tmp_path, unit_id).write_text(json.dumps(
            {"prompt_sha256": digest, "usage": {"output_tokens_details": {"thinking_tokens": 12}}}))
    assert v7_haiku_thinking(v7_answers) == {"70m:real::1": 12}


def test_thinking_split_counts_zero_thinking_and_splits_labels():
    answers, v7_answers = fake({"sonnet-E3": {"real": "A", POOL: "C", "pooled 90/10": "C"}})
    rows = unit_rows(answers, baselines(v7_answers, None))
    real = [r["id"] for r in rows if r["group"] == "real"]
    tokens = {r["id"]: 0 if r["id"] in real else 30 for r in rows}
    out = thinking_split(rows, {"sonnet-E3": tokens, "sonnet-v7": {}})
    assert set(out) == {"sonnet-E3"} and abs(out["sonnet-E3"]["no thinking"] - 1 / 3) < 1e-9
    split = out["sonnet-E3"]["one main job by thinking"]
    assert split["real"] == {"no thinking": {"units": len(real), "one main job": 1.0}, "thinking": {"units": 0, "one main job": None}}
    assert split[POOL]["thinking"] == {"units": len(real), "one main job": 0.0}


def test_odds_ratios_and_swapped_auc():
    arms = {f"haiku-{r}": {"one_main_job": {"real": {"value": v}, POOL: {"value": p}}}
            for r, v, p in (("v7", 0.5, 0.0), ("E3", 0.75, 0.5), ("E4", 0.5, 0.5))}
    out = odds_ratios(arms)
    assert out["haiku: E3 over v7"] == {"real": 3.0, POOL: None} and out["haiku: E4 over v7"]["real"] == 1.0
    answers, v7_answers = fake({"haiku-E3": {"real": "B", POOL: "C"}, "sonnet-E3": {"real": "A", POOL: "B"}})
    assert swapped_auc(unit_rows(answers, baselines(v7_answers, None))) == {"haiku-E3": 0.0, "sonnet-E3": 1.0}


def fake_concurrent(rows: list[dict], labels: dict, thinking: dict) -> dict:
    """Addendum answers: labels and thinking map arm -> group -> value, for real neurons and 50/50 pools."""
    out = {"arms": {arm: {} for arm in ADDENDUM_ARMS}, "spend_usd": 0.5}
    for arm, by_group in labels.items():
        for r in rows:
            if r["group"] in by_group:
                out["arms"][arm][r["id"]] = {"label": by_group[r["group"]], "thinking_tokens": thinking[arm][r["group"]],
                                             "models": ["m"], "cost_usd": 0.001}
    return out


def test_concurrent_addendum_tests_and_drift_answer():
    rng = np.random.default_rng(0)
    answers, v7_answers = fake({"haiku-E3": {"real": "A", POOL: "C"}, "sonnet-E3": {"real": "A", POOL: "C"}})
    base = baselines(v7_answers, None)  # Haiku's v7-rubric baseline is C everywhere; Sonnet has none
    rows = unit_rows(answers, base)
    main = {"haiku-E3": versus_v7(rows, "haiku", "E3", rng)}
    assert main["haiku-E3"]["verdict"] == SELECTIVE
    same = {"real": "A", POOL: "C"}
    concurrent = fake_concurrent(rows, {arm: same for arm in ADDENDUM_ARMS},
                                 {arm: {"real": 0, POOL: 40} for arm in ADDENDUM_ARMS})
    out = concurrent_addendum(rows, concurrent, base, main, rng)
    haiku = out["versus_v7"]["haiku"]
    assert haiku["rise_real"]["value"] == 0.0 and haiku["verdict"] == NO_RISE
    assert out["label_auc_difference"]["haiku"]["value"] == 0.0
    assert out["drift"]["haiku"]["session_v7"]["real"]["value"] == 1.0  # A now, C in the earlier session
    assert out["drift"]["haiku"]["session_E3"]["real"]["value"] == 0.0
    assert out["drift"]["haiku"]["rise_gap"]["real"]["interval"] == [1.0, 1.0]
    assert out["answer"]["haiku"] == {"main verdict": SELECTIVE, "concurrent verdict": NO_RISE,
                                      "verdict": NOT_REPRODUCED, "session_dependent": True}
    assert out["drift"]["sonnet"].startswith("not run") and "sonnet" not in out["answer"]
    no_thinking = out["arms"]["sonnet-E3"]["no_thinking"]
    assert no_thinking["real"]["value"] == 1.0 and no_thinking[POOL]["value"] == 0.0
    assert abs(no_thinking["all"]["value"] - 0.5) < 1e-9  # as many real neurons as pools in the fake
    assert out["arms"]["haiku-v7"]["one_main_job_by_thinking"]["real"]["no thinking"]["one main job"] == 1.0
    assert set(out["arms"]["haiku-v7"]["one_main_job"]) == {"real", POOL}
    assert out["usage"]["haiku-E3"]["readings"] == len(MODELS) * 8


def test_concurrent_addendum_reproduces_a_verdict_read_in_the_same_session():
    rng = np.random.default_rng(0)
    answers, v7_answers = fake({"haiku-E3": {"real": "A", POOL: "C"}})
    base = baselines(v7_answers, None)
    rows = unit_rows(answers, base)
    main = {"haiku-E3": versus_v7(rows, "haiku", "E3", rng)}
    labels = {"haiku-E3": {"real": "A", POOL: "C"}, "haiku-v7": {"real": "C", POOL: "C"},
              "sonnet-E3": {"real": "A", POOL: "C"}, "sonnet-v7": {"real": "C", POOL: "C"}}
    concurrent = fake_concurrent(rows, labels, {arm: {"real": 5, POOL: 5} for arm in ADDENDUM_ARMS})
    out = concurrent_addendum(rows, concurrent, base, main, rng)
    assert out["answer"]["haiku"]["verdict"] == REPRODUCED and not out["answer"]["haiku"]["session_dependent"]
    assert out["arms"]["haiku-E3"]["no_thinking"]["all"]["value"] == 0.0
