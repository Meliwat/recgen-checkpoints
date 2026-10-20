from recgen import rubric_v8
from recgen.rubric_v8 import snippet_line
from recgen.rubric_v16 import E3_DECISION, V7_DECISION
from recgen.rubric_v21 import (
    E3, RETEST_PER_SIZE, V7_RUBRIC, agent_instructions, agent_prompt, as_v8_prompt, infrastructure_failure,
    reading_order_a, retest_ids_a, separates,
)


def test_agent_prompt_differs_from_v8s_only_in_the_decision_paragraph():
    lines = [snippet_line(1, 9.96, "a [[b]] c"), snippet_line(2, 3.04, "d [[e]] f")]
    text, v8 = agent_prompt(lines), rubric_v8.prompt(lines)
    assert E3_DECISION in text and V7_DECISION not in text and as_v8_prompt(text) == v8
    assert "NOT YET UNDERSTOOD" in text and "UNRELATED" not in text and "8 tool calls" in text
    assert '"confidence"' in text and "1. (10.0) a [[b]] c" in text
    assert not any(word in agent_instructions().lower() for word in ("pool", "synthetic", "mixture", "two neurons"))


def test_e3_labels_parse_with_v8s_parser():
    assert rubric_v8.parse_answer('{"label": "C", "patterns": ["x"], "confidence": "low"}')["label"] == "C"
    assert rubric_v8.parse_answer('{"label": "D", "patterns": []}') is None


def test_reading_order_a_pairs_each_units_rubrics_and_is_seeded():
    order = reading_order_a(50, 0)
    assert order == reading_order_a(50, 0) and order != reading_order_a(50, 0, purpose=1)
    assert sorted(order) == sorted((i, r) for i in range(50) for r in (V7_RUBRIC, E3))
    assert all(order[k][0] == order[k + 1][0] and order[k][1] != order[k + 1][1] for k in range(0, 100, 2))
    firsts = [order[k][1] for k in range(0, 100, 2)]
    assert 10 < firsts.count(E3) < 40  # the coin flip mixes which rubric comes first
    assert [order[k][0] for k in range(0, 100, 2)] != list(range(50))


def test_retest_ids_a_are_a_seeded_draw_per_size():
    units = [{"id": f"70m:real::{i}"} for i in range(150)]
    again = retest_ids_a(units, "70m")
    assert len(again) == RETEST_PER_SIZE == len(set(again)) and again == retest_ids_a(units, "70m")
    assert again != retest_ids_a(units, "2.8b")


def test_separates_needs_the_interval_above_chance():
    assert separates([0.55, 0.7]) and not separates([0.48, 0.7]) and not separates([0.5, 0.6])


def test_infrastructure_failures_are_v14s():
    labeled = {"label": "A", "attempts": [{"error": "timeout"}]}
    agent = {"label": None, "attempts": [{"error": "no valid label", "reply": "I think"}, {"error": "error_max_turns"}]}
    limit = {"label": None, "attempts": [{"error": "no valid label", "reply": "You've hit your limit"}]}
    cli = {"label": None, "attempts": [{"error": "no result from the CLI"}]}
    assert not infrastructure_failure(labeled) and not infrastructure_failure(agent)
    assert infrastructure_failure(limit) and infrastructure_failure(cli)
