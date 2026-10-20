import numpy as np

from recgen.analyze_v8 import gain, one_job, per_model, retest, tool_summary, unit_rows
from recgen.rubric_v8 import AGENT_MODEL, MODELS, SECOND_AGENT_MODEL


def fake_inputs(haiku_agent_label=None):
    """Per model: 4 real neurons and 4 pools. One-shot Haiku calls everything A except one pool; the
    Haiku agent calls pools C and real neurons A, except one real neuron it never labeled."""
    v7_units, answers = {}, {arm: {"first": {}, "retest": {}} for arm in (AGENT_MODEL, SECOND_AGENT_MODEL)}
    for model in MODELS:
        v7_units[model] = []
        for i in range(8):
            group = "real" if i < 4 else "pooled 50/50"
            unit_id = f"{model}:{'real' if i < 4 else 'pooled'}:{'' if i < 4 else '50/50'}:{i}"
            v7_units[model].append({"id": unit_id, "model": model, "group": group, "neuron_ids": [i],
                                    "shared_window": False, "minilm": False, "network": i >= 6,
                                    "llm_label": "C" if i == 7 else "A"})
            label = "C" if group != "real" else ("A" if i else haiku_agent_label)
            answers[AGENT_MODEL]["first"][unit_id] = {"label": label}
            answers[SECOND_AGENT_MODEL]["first"][unit_id] = {"label": "B" if group != "real" else "A"}
        first_id = v7_units[model][4]["id"]
        answers[AGENT_MODEL]["retest"][first_id] = {"label": "C" if model != "70m" else "A"}
        answers[SECOND_AGENT_MODEL]["retest"][first_id] = {"label": "B"}
    return answers, v7_units


def test_unit_rows_turn_labels_into_both_flags_and_keep_missing_labels_missing():
    answers, v7_units = fake_inputs()
    rows = unit_rows(answers, v7_units)
    assert all(len(rows[m]) == 8 for m in MODELS)
    first, pool = rows["70m"][0], rows["70m"][4]
    assert first["flag"]["haiku agent"] is None and first["not_one_job"]["haiku agent"] is None
    assert pool["flag"]["haiku agent"] and pool["not_one_job"]["sonnet agent"] and not pool["flag"]["sonnet agent"]
    assert pool["flag"]["network"] is False and rows["70m"][7]["flag"]["haiku"] is True


def test_per_model_keeps_units_every_reader_labeled():
    answers, v7_units = fake_inputs()
    rows = unit_rows(answers, v7_units)
    (agent,), = [per_model(rows, "flag", ("haiku agent",))]
    pooled, real = agent[0]
    assert len(pooled) == 4 and len(real) == 3  # the unlabeled real neuron is dropped
    agent, haiku = per_model(rows, "flag", ("haiku agent", "haiku"))
    assert len(haiku[0][1]) == 3  # and dropped for the paired reader too


def test_a_reader_that_flags_every_pool_and_no_real_neuron_is_much_better():
    answers, v7_units = fake_inputs()
    result = gain(unit_rows(answers, v7_units), "flag", "haiku agent", "haiku", np.random.default_rng(0))
    assert result["haiku agent excess"][0] == 1.0 and np.isclose(result["haiku excess"][0], 0.25)
    assert np.isclose(result["gain"], 0.75) and result["verdict"] == "much better"
    # On the secondary measure the Sonnet agent's B labels count too.
    result = gain(unit_rows(answers, v7_units), "not_one_job", "sonnet agent", "haiku", np.random.default_rng(0))
    assert np.isclose(result["gain"], 0.75)


def test_one_job_shares_and_paired_differences():
    answers, v7_units = fake_inputs(haiku_agent_label="C")
    out = one_job(unit_rows(answers, v7_units), np.random.default_rng(0))
    assert out["haiku"]["share"] == 1.0 and out["haiku agent"]["share"] == 0.75
    assert out["haiku agent minus haiku"]["difference"] == -0.25


def test_retest_counts_same_labels():
    answers, _ = fake_inputs()
    out = retest(answers, np.random.default_rng(0))
    assert out["haiku agent"]["units"] == 3 and out["haiku agent"]["same_label"] == round(2 / 3, 3)
    assert out["sonnet agent"]["same_label"] == 1.0


def test_tool_summary_shows_ranks_and_each_texts_strongest_value():
    assert tool_summary({"tool": "examples", "arguments": {"start": 51, "count": 20}, "answer": ""}) == "examples 51-70"
    call = {"tool": "activation", "arguments": {"texts": ["a cat", "a dog"]},
            "answer": "Text 1, strongest 7.1 at [[ cat]]: a[[ cat|7.1]]\nText 2, strongest 0.2 at [[ dog]]: a dog"}
    assert tool_summary(call) == 'activation on "a cat" (7.1); "a dog" (0.2)'
