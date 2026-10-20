import numpy as np

from recgen.rubric_v15 import (
    DETECTS, INCONCLUSIVE, MIN_DRIVEN, N_HELD_OUT, NO_DETECTION, PER_STRATUM, RANDOM_STRATUM, STRATA, TOP_STRATA,
    coverage, detection_auc, explainer_prompt, group_difference, has_disjunction_marker, held_out_ranks, judge_prompt,
    parse_explanation, parse_judgment, parse_predictions, pearson, separation_auc, simulator_prompt, spearman,
    unit_scores, verdict,
)


def test_held_out_ranks_are_seeded_stratified_and_never_shown():
    picks = held_out_ranks("70m:real::3")
    assert picks == held_out_ranks("70m:real::3") and picks != held_out_ranks("70m:real::4")
    assert len(picks) == N_HELD_OUT == len({r for r, _ in picks})
    for name, (low, high) in STRATA.items():
        ranks = [r for r, s in picks if s == name]
        assert len(ranks) == PER_STRATUM and all(low <= r <= high for r in ranks)
    assert min(r for r, _ in picks) > 50
    assert [s for _, s in picks] != sorted(s for _, s in picks)  # shuffled, not grouped by stratum


def test_held_out_ranks_stay_inside_a_shorter_list():
    assert max(r for r, _ in held_out_ranks("70m:pooled:50/50:1-2", n_ranked=3000)) <= 3000


def test_explainer_prompt_keeps_v8s_snippet_paragraph_and_asks_for_no_label():
    text = explainer_prompt(["1. (10.0) a [[b]] c", "2. (3.0) d [[e]] f"])
    assert text.startswith("Below are 2 text snippets") and "10 is its strongest activation over 4,096" in text
    assert '{"explanation": "..."}' in text and "1. (10.0) a [[b]] c\n2. (3.0) d [[e]] f" in text
    assert "ONE JOB" not in text and "UNRELATED" not in text and "label" not in text


def test_simulator_and_judge_prompts_show_the_explanation_only():
    text = simulator_prompt("a {brace} explanation", ["a [[b]] c", "d [[e]] f"])
    assert "Explanation: a {brace} explanation" in text and "list of 2 numbers" in text
    assert text.endswith("1. a [[b]] c\n2. d [[e]] f") and "(" not in text.split("Snippets:")[1]
    assert '"x or y"' in judge_prompt("x or y") and '"unrelated_patterns": true' in judge_prompt("x")


def test_parsers_accept_only_strict_valid_replies():
    assert parse_explanation('Sure.\n```json\n{"explanation": " Commas after dates. "}\n```') == "Commas after dates."
    assert parse_explanation('{"explanation": ""}') is None and parse_explanation("no json") is None
    assert parse_predictions("[1, 2.5, 10]", n=3) == [1.0, 2.5, 10.0]
    assert parse_predictions("```json\n[0, 0, 7]\n```", n=3) == [0.0, 0.0, 7.0]
    for bad in ("[1, 2]", "[1, 2, 11]", "[1, -1, 2]", '[1, "2", 3]', "[true, 1, 2]", "[1, 2, 3", "none"):
        assert parse_predictions(bad, n=3) is None
    assert parse_judgment('{"unrelated_patterns": false}') is False and parse_judgment('{"unrelated_patterns": "no"}') is None


def test_scores_handle_constant_predictions_and_rank_top_strata():
    strata = [TOP_STRATA[0]] * 2 + [TOP_STRATA[1]] * 2 + [RANDOM_STRATUM] * 2
    true = [8, 7, 4, 3, 1, 0]
    assert np.isclose(pearson(true, true), 1) and np.isclose(spearman(true[::-1], true), -1)
    assert pearson([5] * 6, true) == 0 and spearman([5] * 6, true) == 0
    assert detection_auc([9, 9, 5, 5, 1, 1], strata) == 1 and detection_auc([5] * 6, strata) == 0.5
    scores = unit_scores([5] * 6, true, strata)
    assert scores["constant"] and scores["detection_auc"] == 0.5


def test_group_difference_and_separation_auc_average_over_sizes():
    rng = np.random.default_rng(0)
    per_model = [(np.array([0.2, 0.3, 0.4]), np.array([0.6, 0.7, 0.8])), (np.array([0.5, 0.5]), np.array([0.5, 0.5]))]
    point, interval = group_difference(per_model, rng)
    assert np.isclose(point, (-0.4 + 0) / 2) and interval[0] <= point <= interval[1]
    point, interval = separation_auc([(b, a) for a, b in per_model], rng)
    assert np.isclose(point, (1 + 0.5) / 2) and interval[0] <= point <= interval[1]


def test_verdict_needs_both_conditions_and_else_checks_equivalence():
    assert verdict([-0.2, -0.05], [0.55, 0.8]) == DETECTS
    assert verdict([-0.2, -0.05], [0.45, 0.8]) == INCONCLUSIVE
    assert verdict([-0.08, 0.05], [0.4, 0.6]) == NO_DETECTION
    assert verdict([-0.15, 0.05], [0.4, 0.6]) == INCONCLUSIVE


def test_coverage_separates_the_two_neurons_examples():
    strata = [TOP_STRATA[0]] * 12 + [RANDOM_STRATUM] * 4
    drivers = [0] * 7 + [1] * 5 + [0, 1, 0, 1]
    predicted = [9] * 7 + [0] * 5 + [1, 1, 1, 1]  # the explanation covers neuron 0 only
    out = coverage(predicted, strata, drivers)
    assert out["majority examples"] == 7 and out["minority examples"] == 5 == MIN_DRIVEN
    assert out["majority auc"] == 1 and out["minority auc"] == 0
    assert coverage(predicted, strata, [0] * 9 + [1] * 3 + [0] * 4)["minority auc"] is None


def test_disjunction_markers_are_whole_words():
    assert has_disjunction_marker("Dates or times") and has_disjunction_marker("commas; also code")
    assert not has_disjunction_marker("Ordinal numbers before a noun") and not has_disjunction_marker("Doors")
