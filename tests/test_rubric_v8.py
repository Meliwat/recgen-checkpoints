import numpy as np

from recgen.rubric_v8 import (
    BETTER, MUCH_BETTER, N_POOLED, N_REAL, N_RELATED, NO_DIFFERENCE, NORMALIZING_RANK, WORSE, choose, excess,
    gain_verdict, normalizers, paired_gain, parse_answer, prompt, ranked_rest, retest_ids, snippet_line, unit_values,
)


def fake_units(model: str) -> list[dict]:
    kinds = [("real", None)] * 100 + [("pooled", mix) for mix in ("50/50", "70/30", "90/10") for _ in range(50)]
    kinds += [("related", "50/50")] * 50
    return [{"id": f"{model}:{kind}:{mix or ''}:{i}", "kind": kind, "mix": mix} for i, (kind, mix) in enumerate(kinds)]


def test_choose_draws_seeded_subsets_of_v7s_units_without_other_mixes():
    units = choose(fake_units("410m"), "410m")
    assert units == choose(fake_units("410m"), "410m")
    kinds = [(u["kind"], u["mix"]) for u in units]
    assert kinds.count(("real", None)) == N_REAL and kinds.count(("pooled", "50/50")) == N_POOLED
    assert kinds.count(("related", "50/50")) == N_RELATED and len(units) == N_REAL + N_POOLED + N_RELATED
    assert len({u["id"] for u in units}) == len(units)
    assert [u["id"].split(":")[-1] for u in units] != [u["id"].split(":")[-1] for u in choose(fake_units("2.8b"), "2.8b")]


def test_retest_ids_are_v8_units():
    units = choose(fake_units("70m"), "70m")
    again = retest_ids(units, "70m")
    assert len(again) == 10 and set(again) <= {u["id"] for u in units}


def test_normalizers_take_the_25th_strongest_window():
    window_max = np.stack([np.arange(100.0), 2 * np.arange(100.0)], axis=1)
    assert np.allclose(normalizers(window_max), [100 - NORMALIZING_RANK, 2 * (100 - NORMALIZING_RANK)])


def test_a_pool_is_the_larger_normalized_neuron():
    norms = np.array([2.0, 4.0])
    acts = np.array([[1.0, 1.0], [0.0, 6.0], [3.0, 2.0]])
    assert np.allclose(unit_values(acts, norms), [0.5, 1.5, 1.5])
    assert np.allclose(unit_values(np.array([[3.0]]), np.array([2.0])), [1.5])


def test_a_50_50_pool_ranks_both_neurons_top_25_first():
    rng = np.random.default_rng(0)
    window_max = np.stack([rng.gamma(2, 1, 1000), 5 * rng.gamma(4, 1, 1000)], axis=1)
    values = unit_values(window_max, normalizers(window_max))
    top = set(np.argsort(-values)[:2 * NORMALIZING_RANK])
    expected = set(np.argsort(-window_max[:, 0])[:NORMALIZING_RANK]) | set(np.argsort(-window_max[:, 1])[:NORMALIZING_RANK])
    assert len(expected) < 2 * NORMALIZING_RANK or top == expected


def test_ranked_rest_skips_shown_windows_and_sorts_strongest_first():
    values = np.array([0.1, 0.9, 0.5, 0.9, 0.3])
    assert list(ranked_rest(values, {2})) == [1, 3, 4, 0]


def test_prompt_shows_values_tools_and_v7s_labels():
    text = prompt([snippet_line(1, 9.96, "a [[b]] c"), snippet_line(2, 3.04, "d [[e]] f")])
    assert "2 text snippets" in text and "1. (10.0) a [[b]] c" in text and "2. (3.0) d [[e]] f" in text
    assert "C) UNRELATED JOBS" in text and "8 tool calls" in text and '"confidence"' in text


def test_parse_answer_takes_the_last_valid_json_after_reasoning():
    reply = 'Testing {"x": 1} first. Final: {"label": "C", "patterns": ["dates {like 1990}", "code"], "confidence": "high"}'
    assert parse_answer(reply) == {"label": "C", "patterns": ["dates {like 1990}", "code"], "confidence": "high"}
    assert parse_answer('{"label": "B", "patterns": ["x"], "confidence": "very"}')["confidence"] is None
    assert parse_answer('{"label": "D"}') is None and parse_answer("no json") is None


def test_excess_averages_models():
    per_model = [(np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0])), (np.array([1, 1]), np.array([1, 0]))]
    assert np.isclose(excess(per_model), (0.5 + 0.5) / 2)


def test_paired_gain_is_zero_for_identical_readers_and_positive_for_a_better_one():
    rng = np.random.default_rng(1)
    reader = [(rng.random(40) < 0.6, rng.random(40) < 0.4) for _ in range(3)]
    point, interval, _, _ = paired_gain(reader, reader, np.random.default_rng(2))
    assert point == 0 and interval == [0.0, 0.0]
    better = [(np.ones(40, bool), np.zeros(40, bool)) for _ in range(3)]
    point, interval, first, second = paired_gain(better, reader, np.random.default_rng(3))
    assert point > 0.5 and interval[0] > 0 and first == [1.0, 1.0] and second[0] < second[1]


def test_gain_verdict_needs_the_interval_above_zero_and_the_margin_for_much_better():
    assert gain_verdict(0.2, [0.05, 0.35]) == MUCH_BETTER
    assert gain_verdict(0.1, [0.02, 0.2]) == BETTER
    assert gain_verdict(0.2, [-0.01, 0.4]) == NO_DIFFERENCE
    assert gain_verdict(-0.2, [-0.3, -0.1]) == WORSE
