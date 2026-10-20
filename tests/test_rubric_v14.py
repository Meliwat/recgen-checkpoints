from recgen.rubric_v8 import MODELS, SECOND_AGENT_MODEL
from recgen.rubric_v14 import (
    AGENT_MODEL, AGENT_NAME, PILOT_MODEL, PILOT_UNITS, PRIMARY, SECONDARY, infrastructure_failure, pilot_units,
)


def reading(label=None, *attempts):
    return {"label": label, "attempts": [{"error": e, "reply": r} for e, r in attempts]}


def test_the_arm_is_opus_by_full_id_and_the_primary_comparison_is_against_v8s_sonnet_agent():
    assert AGENT_MODEL == "claude-opus-5-5" and SECOND_AGENT_MODEL == "sonnet"
    assert PRIMARY == (AGENT_NAME, "sonnet agent")
    assert {b for _, b in SECONDARY} == {"haiku", "haiku agent"} and all(a == AGENT_NAME for a, _ in SECONDARY)


def test_pilot_units_are_outside_v8s_sample_and_take_the_first_of_each_kind():
    assert PILOT_MODEL not in MODELS
    units = [{"id": i, "kind": k, "mix": m} for i, (k, m) in enumerate(
        [("real", None), ("real", None), ("pooled", "50/50"), ("pooled", "50/50"), ("related", "50/50")])]
    assert [u["id"] for u in pilot_units(units)] == [0, 2]
    assert len(PILOT_UNITS) == 2


def test_a_labeled_reading_is_never_an_infrastructure_failure():
    assert not infrastructure_failure(reading("A", ("Command timed out", ""), (None, "")))


def test_the_agents_own_failures_are_final():
    assert not infrastructure_failure(reading(None, ("no valid label", "I think it is mixed"),
                                              ("error_max_turns", ""), ("error_max_budget_usd", "")))


def test_cli_errors_timeouts_and_missing_tools_are_read_again():
    assert infrastructure_failure(reading(None, ("no valid label", "hmm"), ("probe tools not offered", "")))
    assert infrastructure_failure(reading(None, ("Command '['claude']' timed out after 900 seconds", "")))
    assert infrastructure_failure(reading(None, ("success", "")))  # is_error with an unexpected subtype


def test_a_usage_limit_reply_without_a_label_is_read_again():
    assert infrastructure_failure(reading(None, ("no valid label", "Claude AI usage limit reached|1760000000")))
    assert infrastructure_failure(reading(None, ("no valid label", "You've hit your limit, resets 3am")))
    assert infrastructure_failure(reading(None, ("no valid label", "API Error: 529 Overloaded")))


def test_a_reading_the_spend_guard_stopped_is_not_a_failure():
    assert not infrastructure_failure(reading(None, ("spend cap reached", "")))
