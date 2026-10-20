from recgen import rubric_v7, rubric_v16
from recgen.rubric_v16_addendum import (
    ARMS, NOT_REPRODUCED, REPRODUCED, SPEND_CAP_USD, drift_answer, make_prompt, reading_order,
)

SNIPPETS = ["a [[b]] c", "d [[e]] f"]


def test_prompts_are_v7s_and_v16s_exactly():
    assert make_prompt("v7")(SNIPPETS) == rubric_v7.prompt(SNIPPETS)
    assert make_prompt("E3")(SNIPPETS) == rubric_v16.prompt("E3", SNIPPETS)


def test_two_readers_by_two_rubrics_and_the_cap():
    assert ARMS == {"haiku-v7": ("haiku", "v7"), "haiku-E3": ("haiku", "E3"),
                    "sonnet-v7": ("sonnet", "v7"), "sonnet-E3": ("sonnet", "E3")}
    assert SPEND_CAP_USD == 40.0


def test_reading_order_is_one_seeded_permutation_that_mixes_arms():
    order = reading_order(400, 0)
    assert sorted(order) == list(range(400)) and order == reading_order(400, 0) and order != reading_order(400, 1)
    arms_of_first_40 = {i % 4 for i in order[:40]}  # jobs listed arm by arm within each unit
    assert arms_of_first_40 == {0, 1, 2, 3}


def test_drift_answer():
    assert drift_answer("a", "a", [-0.05, 0.02]) == {"verdict": REPRODUCED, "session_dependent": False}
    assert drift_answer("a", "b", [0.01, 0.2]) == {"verdict": NOT_REPRODUCED, "session_dependent": True}
