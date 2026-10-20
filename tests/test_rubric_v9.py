import numpy as np

from recgen.rubric_v9 import (
    MINORITY_BINS, PEAK, TOP_50, mean_excess, minority_bin, minority_count, scale, union_top,
)


def test_scale_is_the_median_of_the_top_50_or_the_peak():
    window_max = np.arange(100, dtype=float)  # top 50 are 50..99, median 74.5
    assert scale(window_max, TOP_50) == 74.5 and scale(window_max, PEAK) == 99.0


def test_union_top_takes_each_window_from_the_stronger_scaled_neuron():
    max_a, max_b = np.array([4.0, 1.0, 3.0, 0.0]), np.array([0.0, 10.0, 2.0, 1.0])
    pos_a, pos_b = np.array([40, 41, 42, 43]), np.array([50, 51, 52, 53])
    top = union_top(max_a, pos_a, max_b, pos_b, scale_a=4.0, scale_b=10.0, k=3)
    # scaled: a = [1, .25, .75, 0], b = [0, 1, .2, .1]; union = [1, 1, .75, .1]
    assert top == [(0, 40, 0), (1, 51, 1), (2, 42, 0)]


def test_union_top_windows_are_among_each_neurons_own_top_windows():
    rng = np.random.default_rng(0)
    max_a, max_b = rng.gamma(2, size=400), rng.gamma(2, size=400) * 3
    pos = np.zeros(400, dtype=int)
    top = union_top(max_a, pos, max_b, pos, scale_a=2.0, scale_b=7.0, k=50)
    own = set(np.argsort(-max_a)[:50]) | set(np.argsort(-max_b)[:50])
    assert all(w in own for w, _, _ in top)


def test_minority_count_and_bins():
    assert minority_count([0] * 45 + [1] * 5) == 5 and minority_count([1] * 50) == 0
    assert [minority_bin(c) for c in (0, 4, 5, 14, 15, 25)] == [0, 0, 1, 1, 2, 2]
    assert MINORITY_BINS[-1][1] > 25


def test_mean_excess_averages_models_and_brackets_the_point():
    groups = [(np.array([1.0] * 6 + [0.0] * 4), np.array([1.0] * 2 + [0.0] * 8))] * 3
    point, (low, high) = mean_excess(groups, np.random.default_rng(0))
    assert abs(point - 0.4) < 1e-9 and low < 0.4 < high
