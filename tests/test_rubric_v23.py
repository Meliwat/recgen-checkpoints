import numpy as np

from recgen.rubric_v23 import (
    MORE_JOBS, NO_GRADIENT, PER_QUARTILE, UNCLEAR, WEAK_SHARE_MORE, control_reading, facet_count_strata,
    least_overlapping, neuron_measures, paired_difference, quartiles, reading, rng_for, select, spearman,
    top_minus_rest,
)


def test_quartiles_are_v12s():
    q = quartiles(np.arange(8.0))
    assert list(q) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_select_draws_per_quartile_from_eligible_only_with_a_fixed_seed():
    quartile = np.repeat(np.arange(4), 100)
    eligible = np.ones(400, bool)
    eligible[::2] = False
    eligible[300:] = False
    eligible[[301, 303, 305]] = True  # only 3 eligible in the top quartile
    a, b = select(quartile, eligible), select(quartile, eligible)
    assert np.array_equal(a, b) and eligible[a].all() and list(a) == sorted(a)
    assert [int(np.sum(quartile[a] == q)) for q in range(4)] == [PER_QUARTILE] * 3 + [3]


def test_least_overlapping_breaks_ties_by_pair_order():
    assert least_overlapping({"0-1": 0.1, "0-2": 0.0, "1-2": 0.0}) == "0-2"
    assert least_overlapping({"1-10": 0.0, "1-2": 0.0}) == "1-2"


def test_neuron_measures_follow_v2s_rule():
    m = neuron_measures({"0-1": 0.3, "0-2": 0.05, "1-2": 0.1}, 0.1, reader_least="0-1")
    assert m["least_pair"] == "0-2" and m["least_unrelated"]
    assert m["share_unrelated"] == 2 / 3  # 0.1 is at the threshold, so not adjacent
    assert abs(m["margin"] + 0.05) < 1e-12 and not m["reader_least_unrelated"]
    assert not neuron_measures({"0-1": 0.2}, 0.1, "0-1")["least_unrelated"]


def test_top_minus_rest_and_its_interval():
    quartile = np.repeat(np.arange(4), 50)
    values = (quartile == 3).astype(float)
    out = top_minus_rest(quartile, values, rng_for(0))
    assert out["value"] == 1.0 and out["interval"] == [1.0, 1.0]


def test_stratified_difference_removes_a_confound():
    # Within each stratum top and rest are equal; strata differ in level and in their share of top neurons.
    quartile = np.array([3, 0, 0, 0, 3, 3, 3, 0] * 20)
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 20)
    values = strata.astype(float)
    assert top_minus_rest(quartile, values, rng_for(0))["value"] > 0.4
    stratified = top_minus_rest(quartile, values, rng_for(0), strata)
    assert stratified["value"] == 0 and stratified["interval"] == [0.0, 0.0]


def test_paired_difference_and_spearman():
    quartile = np.repeat(np.arange(4), 50)
    a = (quartile == 3).astype(float)
    assert paired_difference(quartile, a, a, rng_for(1))["value"] == 0
    assert paired_difference(quartile, a, 0 * a, rng_for(1))["interval"] == [1.0, 1.0]
    peaks = np.arange(200.0)
    assert spearman(quartile, peaks, peaks, rng_for(2))["value"] == 1.0


def test_reading_order():
    below, above, around = [-0.3, -0.1], [0.1, 0.3], [-0.1, 0.1]
    assert reading(below, above, around) == MORE_JOBS
    assert reading(above, below, around) == WEAK_SHARE_MORE
    assert reading(around, around, above) == NO_GRADIENT
    assert reading(below, around, above) == UNCLEAR  # P1 alone says more jobs, so not "no gradient"
    assert reading(around, around, around) == UNCLEAR


def test_control_reading_and_facet_count_strata():
    assert control_reading([-0.3, -0.1], [-0.2, -0.05]) == "holds within strata"
    assert control_reading([-0.3, -0.1], [-0.2, 0.05]) == "may be explained by the stratifying variable"
    assert control_reading([0.1, 0.3], [0.05, 0.2]) == "holds within strata"
    assert control_reading([-0.1, 0.1], [-0.2, -0.1]).startswith("nothing to explain")
    assert list(facet_count_strata(np.array([2, 3, 4, 6]))) == [2, 3, 4, 4]
