import numpy as np

from recgen import rubric_v7
from recgen.rubric_v16 import (
    ARMS, INFLATES, LABEL_RANK, LOWERS, NO_RISE, SELECTIVE, auc, bootstrap, changes, instructions, near_essay,
    not_one_main_job, parse_label, prompt, reading_order, unrelated_jobs, verdict,
)


def test_only_the_decision_paragraph_differs_from_v7():
    v7 = rubric_v7.prompt(["a [[b]] c", "d [[e]] f"])
    for rubric in ("E3", "E4"):
        text = prompt(rubric, ["a [[b]] c", "d [[e]] f"])
        head, tail = v7.split("Decide which")[0], v7.split("Reply with JSON only")[1]
        assert text.startswith(head) and text.endswith(tail)
        assert "2 text snippets" in text and "1. a [[b]] c" in text and '{"label": "A", "patterns"' in text
    e3, e4 = instructions("E3"), instructions("E4")
    assert "one main job, often with small related offshoots" in e3 and "NOT YET UNDERSTOOD" in e3
    assert "UNRELATED" not in e3 and "D) TWO OR MORE UNRELATED JOBS" in e4
    assert e4.replace("\nD) TWO OR MORE UNRELATED JOBS: two or more jobs with no clear relation to each other.", "") == e3


def test_prompts_never_mention_pools_or_synthetic_units():
    for rubric in ("E3", "E4"):
        text = instructions(rubric).lower()
        assert not any(word in text for word in ("pool", "synthetic", "mixture", "artificial", "two neurons"))


def test_parse_label_accepts_only_the_rubric_labels():
    assert parse_label('{"label": "C", "patterns": []}', "E3") == "C"
    assert parse_label('{"label": "D", "patterns": ["x", "y"]}', "E3") is None
    assert parse_label('Here:\n```json\n{"label": "D", "patterns": ["x", "y"]}\n```', "E4") == "D"
    assert parse_label("no json", "E4") is None and parse_label('{"label": "E"}', "E4") is None


def test_flags():
    assert not_one_main_job("A") is False and not_one_main_job("C") is True and not_one_main_job(None) is None
    assert unrelated_jobs("D") is True and unrelated_jobs("C") is False and unrelated_jobs(None) is None
    assert LABEL_RANK["E4"] == {"A": 0, "B": 1, "C": 2, "D": 3} and LABEL_RANK["v7"]["C"] == 2
    assert set(ARMS) == {"haiku-E3", "haiku-E4", "sonnet-E3", "sonnet-E4"}


def test_reading_order_puts_main_groups_first_then_retests_and_is_seeded():
    jobs = [("real", False), ("pooled 70/30", False), ("pooled 50/50", False), ("real", True), ("related", False)]
    order = reading_order(jobs, 0)
    assert sorted(order) == list(range(5)) and order == reading_order(jobs, 0)
    assert set(order[:2]) == {0, 2} and order[2] == 3 and set(order[3:]) == {1, 4}


def test_bootstrap_is_paired_across_columns():
    rng = np.random.default_rng(0)
    values = rng.random((50, 1))
    strata = {"a": np.hstack([values, values + 1.0])}
    result = bootstrap(strata, lambda s: float(np.mean(s["a"][:, 1] - s["a"][:, 0])), rng, draws=200)
    assert abs(result["value"] - 1.0) < 1e-9 and np.allclose(result["interval"], [1.0, 1.0])


def test_auc_counts_ties_as_half():
    assert auc(np.array([2, 2]), np.array([0, 0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert auc(np.array([0, 2]), np.array([1])) == 0.5


def test_verdict_rule():
    assert verdict([0.05, 0.2], [-0.1, 0.05]) == INFLATES
    assert verdict([0.05, 0.2], [0.01, 0.2]) == INFLATES
    assert verdict([0.05, 0.2], [-0.3, -0.01]) == SELECTIVE
    assert verdict([-0.02, 0.1], [0.0, 0.1]) == NO_RISE
    assert verdict([-0.2, -0.01], [0.0, 0.1]) == LOWERS


def test_near_essay_and_changes():
    assert near_essay([0.85, 0.92]) and near_essay([0.91, 0.97]) and not near_essay([0.6, 0.7])
    assert changes([0.01, 0.2]) and changes([-0.2, -0.01]) and not changes([-0.1, 0.1])
