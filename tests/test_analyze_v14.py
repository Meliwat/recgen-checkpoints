import numpy as np

from recgen.analyze_v8 import gain, kappas, label_auc, unit_rows
from recgen.analyze_v14 import (
    AGENTS, LLM_READERS, READERS, label_auc_difference, label_changes, label_shares, one_job_shares, prompts_match,
    reading_stats, shared_unrelated, tool_use,
)
from recgen.rubric_v8 import AGENT_MODEL as HAIKU, MODELS, SECOND_AGENT_MODEL as SONNET
from recgen.rubric_v14 import AGENT_MODEL as OPUS, AGENT_NAME


def record(label, calls=0, prompt="p"):
    return {"label": label, "patterns": ["x"], "confidence": "high", "prompt_sha256": prompt, "attempts": [{"error": None}],
            "tools_offered": ["examples", "activation"], "models": ["m"], "cost_usd": 0.1, "input_tokens": 10,
            "output_tokens": 1, "tool_calls": [{"tool": "examples", "arguments": {"start": 51, "count": 5}}] * calls}


def fake_inputs():
    """Per model: 4 real neurons and 4 pools. The Sonnet agent calls pools B and real neurons A; the
    Opus agent calls pools C, two real neurons A and two B; the Haiku agent calls everything A."""
    v7_units, answers = {}, {arm: {"first": {}, "retest": {}} for arm in (HAIKU, SONNET, OPUS)}
    for model in MODELS:
        v7_units[model] = []
        for i in range(8):
            real = i < 4
            unit_id = f"{model}:{'real::' if real else 'pooled:50/50:'}{i}"
            v7_units[model].append({"id": unit_id, "model": model, "group": "real" if real else "pooled 50/50",
                                    "neuron_ids": [i], "shared_window": False, "minilm": False, "network": not real,
                                    "llm_label": "A"})
            answers[HAIKU]["first"][unit_id] = record("A")
            answers[SONNET]["first"][unit_id] = record("A" if real else "B", calls=3)
            answers[OPUS]["first"][unit_id] = record(("A" if i < 2 else "B") if real else "C", calls=4 + i % 3)
    return answers, v7_units


def test_the_opus_agent_gets_its_own_column_and_the_primary_gain_compares_it_with_sonnet():
    answers, v7_units = fake_inputs()
    rows = unit_rows(answers, v7_units, AGENTS)
    assert set(rows["70m"][0]["labels"]) == {"haiku", "haiku agent", "sonnet agent", AGENT_NAME}
    result = gain(rows, "flag", AGENT_NAME, "sonnet agent", np.random.default_rng(0))
    assert result["gain"] == 1.0 and result["verdict"] == "much better"
    # With B or C as the flag, the Sonnet agent's B on pools counts, and the Opus agent's B on real neurons costs it.
    result = gain(rows, "not_one_job", AGENT_NAME, "sonnet agent", np.random.default_rng(0))
    assert np.isclose(result["gain"], -0.5)


def test_one_job_shares_include_the_paired_opus_minus_sonnet_difference():
    answers, v7_units = fake_inputs()
    rows = unit_rows(answers, v7_units, AGENTS)
    real = one_job_shares(rows, "real", np.random.default_rng(0))
    assert real[AGENT_NAME]["share"] == 0.5 and real["sonnet agent"]["share"] == 1.0
    assert real[f"{AGENT_NAME} minus sonnet agent"]["difference"] == -0.5
    assert real[f"{AGENT_NAME} minus haiku"]["difference"] == -0.5
    assert one_job_shares(rows, "pooled 50/50", np.random.default_rng(0))[AGENT_NAME]["share"] == 0.0


def test_kappas_pair_the_opus_agent_with_each_of_the_five_other_readers():
    answers, v7_units = fake_inputs()
    out = kappas(unit_rows(answers, v7_units, AGENTS), {AGENT_NAME: OPUS}, READERS)
    assert sorted(out) == sorted(f"{AGENT_NAME} vs {r}" for r in READERS if r != AGENT_NAME)


def test_prompts_match_compares_with_the_sonnet_agents_prompt_for_each_unit():
    answers, _ = fake_inputs()
    first = next(iter(answers[OPUS]["first"]))
    answers[OPUS]["first"][first]["prompt_sha256"] = "other"
    out = prompts_match(answers)
    assert out == {"same": 23, "compared": 24, "opus readings": 24}


def test_tool_use_reports_mean_and_most_tool_calls_per_agent():
    answers, v7_units = fake_inputs()
    out = tool_use(unit_rows(answers, v7_units, AGENTS), answers)
    assert out["sonnet agent"]["mean tool calls, all first readings"] == 3.0
    assert out[AGENT_NAME]["most tool calls"] == 6
    assert out[AGENT_NAME]["tool use"]["real"]["share reading past rank 50"] == 1.0


def test_label_changes_count_each_pair_of_labels_per_group():
    answers, v7_units = fake_inputs()
    out = label_changes(unit_rows(answers, v7_units, AGENTS), "sonnet agent", AGENT_NAME)
    assert out == {"real": {"A->A": 6, "A->B": 6}, "pooled 50/50": {"B->C": 12}}


def test_tool_use_reports_whether_the_budget_bound():
    answers, v7_units = fake_inputs()
    answers[OPUS]["first"]["70m:real::0"] = record("A", calls=8)
    out = tool_use(unit_rows(answers, v7_units, AGENTS), answers)
    assert out[AGENT_NAME]["share using all 8"] == round(1 / 24, 3) and out["sonnet agent"]["share using all 8"] == 0


def test_label_shares_give_a_or_b_as_one_minus_c_and_the_b_or_c_ceiling():
    answers, v7_units = fake_inputs()
    out = label_shares(unit_rows(answers, v7_units, AGENTS))
    assert out[AGENT_NAME]["real"] == {"A": 0.5, "B": 0.5, "C": 0.0, "B or C": 0.5, "A or B": 1.0,
                                       "B or C per model": {m: 0.5 for m in MODELS}}
    assert out[AGENT_NAME]["pooled 50/50"]["C"] == 1.0 and out[AGENT_NAME]["pooled 50/50"]["A or B"] == 0.0
    # Calling half of real neurons B or C leaves room for at most 50 points of B-or-C detection.
    assert out[AGENT_NAME]["B-or-C ceiling"] == 0.5 and out["sonnet agent"]["B-or-C ceiling"] == 1.0
    for reader in LLM_READERS:
        for group in ("real", "pooled 50/50"):
            assert out[reader][group]["A or B"] == round(1 - out[reader][group]["C"], 3)


def test_label_auc_difference_matches_the_two_readers_aucs_and_is_paired():
    answers, v7_units = fake_inputs()
    rows = unit_rows(answers, v7_units, AGENTS)
    # Opus: pools C, real neurons half A and half B (AUC 1); Sonnet: pools B, real neurons A (AUC 1); Haiku agent: all A (0.5).
    out = label_auc_difference(rows, AGENT_NAME, "haiku agent", np.random.default_rng(0))
    expected = label_auc(rows, AGENT_NAME, np.random.default_rng(0))["mean_over_models"] - label_auc(
        rows, "haiku agent", np.random.default_rng(0))["mean_over_models"]
    assert out["difference"] == round(expected, 3) == 0.5
    # With the Sonnet agent missing two pools per size its AUC varies across draws, but a reader paired
    # with itself differs by exactly 0 in every draw, because both columns are drawn from the same units.
    for model in MODELS:
        for i in (4, 5):
            answers[SONNET]["first"][f"{model}:pooled:50/50:{i}"]["label"] = "A"
    rows = unit_rows(answers, v7_units, AGENTS)
    assert label_auc_difference(rows, "sonnet agent", "sonnet agent", np.random.default_rng(0))["interval"] == [0.0, 0.0]
    low, high = label_auc_difference(rows, AGENT_NAME, "sonnet agent", np.random.default_rng(0))["interval"]
    assert low < high


def test_shared_unrelated_counts_each_readers_c_labels_and_the_overlap():
    answers, v7_units = fake_inputs()
    answers[SONNET]["first"]["70m:pooled:50/50:4"]["label"] = "C"
    answers[SONNET]["first"]["70m:real::2"]["label"] = "C"  # the Opus agent calls this one B
    out = shared_unrelated(unit_rows(answers, v7_units, AGENTS), AGENT_NAME, "sonnet agent")
    assert out["real"] == {f"{AGENT_NAME} C": 0, "sonnet agent C": 1, "both C": 0}
    assert out["pooled 50/50"] == {f"{AGENT_NAME} C": 12, "sonnet agent C": 1, "both C": 1}
    assert out["both groups"] == {f"{AGENT_NAME} C": 12, "sonnet agent C": 2, "both C": 1}


def test_reading_stats_report_the_deepest_rank_and_per_reading_tokens_and_cost():
    answers, _ = fake_inputs()
    answers[OPUS]["retest"]["70m:real::0"] = {**record("A"), "output_tokens": 49, "tool_calls": [
        {"tool": "examples", "arguments": {"start": 3000, "count": 15}}, {"tool": "activation", "arguments": {"texts": ["a"]}}]}
    out = reading_stats(answers)
    assert out[AGENT_NAME] == {"readings": 25, "deepest rank asked for": 3014, "output tokens per reading": 3.0,
                               "cost per reading, USD": 0.1}
    assert out["sonnet agent"]["deepest rank asked for"] == 55 and out["haiku agent"]["deepest rank asked for"] is None
