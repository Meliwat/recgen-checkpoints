import numpy as np

from recgen.analyze_v8 import gain, kappas, unit_rows
from recgen.analyze_v14 import (
    AGENTS, LLM_READERS, READERS, label_changes, label_shares, one_job_shares, prompts_match, tool_use,
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
