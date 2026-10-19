import numpy as np

from recgen.rubric_v5 import (
    AGAINST, FITS, NO_TREND, SAMPLE_SIZE, bootstrap_interval, decide, eligible, layer_index, sample, unrelated_mixed,
)


def test_layer_index_rounds_depth_down():
    assert [layer_index(6, d) for d in (0.25, 0.5, 0.75)] == [1, 3, 4]
    assert [layer_index(32, d) for d in (0.25, 0.5, 0.75)] == [8, 16, 24]


def test_eligible_drops_near_dead_and_dense_neurons():
    rng = np.random.default_rng(0)
    window_max = rng.uniform(0, 0.2, size=(100, 4))
    window_max[0, :] = [2.0, 2.0, 0.5, 2.0]  # neuron 2 peaks at a quarter of the median peak
    window_max[:60, 3] = 1.5  # neuron 3 reaches half its peak in 60% of windows
    assert eligible(window_max).tolist() == [0, 1]


def test_sample_keeps_everything_when_the_layer_is_small_and_is_seeded_otherwise():
    assert sample(np.arange(10), np.random.default_rng(0)).tolist() == list(range(10))
    big = np.arange(SAMPLE_SIZE * 3)
    a, b = sample(big, np.random.default_rng(1)), sample(big, np.random.default_rng(1))
    assert len(a) == SAMPLE_SIZE and np.array_equal(a, b)


def test_unrelated_mixed_needs_two_facets_and_a_dissimilar_pair():
    assert unrelated_mixed(2, 0.05)
    assert not unrelated_mixed(2, 0.3)
    assert not unrelated_mixed(1, None)


def test_bootstrap_interval_brackets_the_share():
    flags = np.array([True] * 30 + [False] * 70)
    low, high = bootstrap_interval(flags, np.random.default_rng(0))
    assert low < 0.3 < high and high - low < 0.25


def test_decide_needs_both_a_monotone_trend_and_separated_ends():
    falling = [0.30, 0.27, 0.22, 0.20, 0.15, 0.10]
    tight = [(s - 0.02, s + 0.02) for s in falling]
    assert decide(falling, tight)[0] == FITS
    assert decide(falling[::-1], tight[::-1])[0] == AGAINST
    wide = [(0.0, 1.0)] * 6
    assert decide(falling, wide)[0] == NO_TREND
    assert decide([0.2, 0.3, 0.1, 0.25, 0.15, 0.2], tight)[0] == NO_TREND
