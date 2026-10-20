import json

import numpy as np

from recgen import analyze_v16
from recgen.analyze_v16 import (
    POOL, analyze, arm_stats, auc_differences, baselines, e4_versus_e3, patterns_of, retest, unit_place, unit_rows,
    v10_sonnet, versus_v7,
)
from recgen.llm_reader import cache_path
from recgen.rubric_v5 import MODELS
from recgen.rubric_v16 import ARMS, INFLATES, SELECTIVE

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


def test_auc_differences_are_paired_per_reader():
    answers, v7_answers = fake({"haiku-E3": {"real": "A", POOL: "C"}, "haiku-E4": {"real": "A", POOL: "A"}},
                               v7={"real": "A", POOL: "C", "pooled 90/10": "C"})
    base = baselines(v7_answers, None)
    out = auc_differences(unit_rows(answers, base), base, np.random.default_rng(0))
    assert out["haiku: E3 minus E4"]["value"] == 0.5 and out["haiku: E3 minus v7"]["interval"] == [0.0, 0.0]
    assert out["haiku: E4 minus v7"]["value"] == -0.5 and not any(k.startswith("sonnet") for k in out)
