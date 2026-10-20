import numpy as np

from recgen.rubric_v8 import INSTRUCTIONS as V8_INSTRUCTIONS, TOOLS as V8_TOOLS
from recgen.rubric_v18 import (
    AUC_MARGIN, BETTER, MAX_TOOL_CALLS, MIN_TOOL_CALLS, MUCH_BETTER, N_LISTED, NO_DIFFERENCE, TOOLS, WORSE, auc_verdict,
    infrastructure_failure, label_auc, paired_auc_difference, parse_answer, prompt, round_robin,
)


def test_prompt_keeps_v8s_snippets_and_definitions_and_adds_the_protocol():
    lines = ["1. (9.0) a [[b]] c", "2. (7.5) d [[e]]"]
    v8 = V8_INSTRUCTIONS.format(n=2, snippets="\n".join(lines))
    ours = prompt(lines)
    head = v8.split("\n\nInvestigate before you decide.")[0]
    assert ours.startswith(head) and ours.endswith("Snippets:\n" + "\n".join(lines))
    assert f"You have {MAX_TOOL_CALLS} tool calls; use at least {MIN_TOOL_CALLS}" in ours
    assert "ranks 51 to 500" in ours and '"description"' in ours and "synthetic" not in ours.lower()


def test_tools_are_v8s_except_the_examples_list_length():
    assert [t["name"] for t in TOOLS] == [t["name"] for t in V8_TOOLS]
    assert TOOLS[1] == V8_TOOLS[1]
    assert f"{N_LISTED:,} windows" in TOOLS[0]["description"] and TOOLS[0]["inputSchema"]["properties"]["start"]["maximum"] == N_LISTED


def test_parse_answer_needs_a_label_and_a_description():
    assert parse_answer('x {"label": "C", "description": " dates; or code ", "confidence": "low"} y') == {
        "label": "C", "description": "dates; or code", "confidence": "low"}
    assert parse_answer('{"label": "A", "patterns": ["p"], "confidence": "high"}') is None
    assert parse_answer('{"label": "D", "description": "x"}') is None
    assert parse_answer('{"label": "A", "description": "  "}') is None
    assert parse_answer('{"label": "B", "description": "x", "confidence": "sure"}')["confidence"] is None


def test_infrastructure_failure_separates_the_cli_from_the_agent():
    agent = {"label": None, "attempts": [{"error": "no valid label", "reply": "I think A"}, {"error": "error_max_turns"}]}
    assert not infrastructure_failure(agent)
    assert infrastructure_failure({"label": None, "attempts": [{"error": "no valid label", "reply": "You've hit your limit"}]})
    assert infrastructure_failure({"label": None, "attempts": [{"error": "Command timed out after 1800 seconds"}]})
    assert not infrastructure_failure({"label": "A", "attempts": [{"error": "CLI failed"}, {"error": None}]})


def test_round_robin_interleaves_kinds_and_keeps_every_unit():
    units = [{"id": f"r{i}", "kind": "real"} for i in range(4)] + [{"id": f"p{i}", "kind": "pooled"} for i in range(4)] + [
        {"id": f"x{i}", "kind": "related"} for i in range(2)]
    order = [u["id"] for u in round_robin(units)]
    assert order[:6] == ["r0", "p0", "x0", "r1", "p1", "x1"] and order[6:] == ["r2", "p2", "r3", "p3"]


def test_label_auc_and_its_paired_difference():
    perfect = [(np.array([2, 2, 2]), np.array([0, 0, 0]))] * 3
    chance = [(np.array([1, 1, 1]), np.array([1, 1, 1]))] * 3
    assert label_auc(perfect) == 1.0 and label_auc(chance) == 0.5
    point, interval, a, b = paired_auc_difference(perfect, chance, np.random.default_rng(0))
    assert point == 0.5 and interval == [0.5, 0.5] and a == [1.0, 1.0] and b == [0.5, 0.5]
    same, same_interval, *_ = paired_auc_difference(perfect, perfect, np.random.default_rng(0))
    assert same == 0 and same_interval == [0.0, 0.0]


def test_auc_verdict_needs_the_margin_for_much_better():
    assert auc_verdict(AUC_MARGIN, [0.01, 0.2]) == MUCH_BETTER
    assert auc_verdict(AUC_MARGIN - 0.01, [0.01, 0.2]) == BETTER
    assert auc_verdict(-0.1, [-0.2, -0.01]) == WORSE
    assert auc_verdict(0.2, [-0.01, 0.4]) == NO_DIFFERENCE
