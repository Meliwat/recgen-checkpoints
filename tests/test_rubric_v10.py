import numpy as np

from recgen.rubric_v6 import cohen_kappa
from recgen.rubric_v10 import (
    BETTER, NO_DIFFERENCE, WORSE, compare, excess, excess_interval, kappa_ceiling, kappa_interval, multiple_jobs,
    paired_difference, paired_kappa_difference, reading_order,
)


def flags(n_true: int, n: int) -> np.ndarray:
    return np.array([True] * n_true + [False] * (n - n_true))


def test_reading_order_puts_the_main_comparison_first_then_retests_then_the_rest():
    jobs = [("related", False), ("real", False), ("pooled 70/30", False), ("pooled 50/50", False),
            ("real", True), ("pooled 90/10", False), ("real", False), ("related", True)]
    order = reading_order(jobs, seed=0)
    assert sorted(order) == list(range(len(jobs))) and order == reading_order(jobs, seed=0)
    assert set(order[:3]) == {1, 3, 6} and set(order[3:5]) == {4, 7} and set(order[5:]) == {0, 2, 5}


def test_multiple_jobs_counts_related_and_unrelated_answers():
    assert [multiple_jobs(x) for x in ("A", "B", "C", None)] == [False, True, True, None]


def test_excess_is_the_mean_over_models_of_pooled_minus_real():
    pooled = [np.array([[1, 1], [1, 0]], float), np.array([[0, 1], [0, 1]], float)]
    real = [np.array([[0, 0], [0, 0]], float), np.array([[0, 1], [0, 0]], float)]
    # reader 0: (1.0 - 0) and (0 - 0) -> 0.5; reader 1: (0.5 - 0) and (1.0 - 0.5) -> 0.5
    assert np.allclose(excess(pooled, real), [0.5, 0.5])


def test_excess_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    out = excess_interval([flags(30, 50)] * 6, [flags(40, 100)] * 6, rng)
    assert abs(out["excess"] - 0.2) < 1e-9 and out["interval"][0] < 0.2 < out["interval"][1]


def test_compare_needs_the_whole_interval_on_one_side_of_zero():
    assert compare(0.01, 0.2) == BETTER and compare(-0.2, -0.01) == WORSE and compare(-0.01, 0.2) == NO_DIFFERENCE


def test_identical_readers_differ_by_exactly_zero():
    rng = np.random.default_rng(0)
    p, r = flags(30, 50), flags(40, 100)
    out = paired_difference([np.stack([p, p], 1)] * 6, [np.stack([r, r], 1)] * 6, rng)
    assert out["difference"] == 0 and out["interval"] == [0.0, 0.0] and out["verdict"] == NO_DIFFERENCE


def test_a_reader_that_separates_pools_beats_haiku_and_the_reverse_is_worse():
    rng = np.random.default_rng(0)
    haiku_pooled, haiku_real = flags(30, 50), flags(40, 100)  # excess +0.2
    reader_pooled, reader_real = flags(45, 50), flags(20, 100)  # excess +0.7
    pooled = [np.stack([haiku_pooled, reader_pooled], 1)] * 6
    real = [np.stack([haiku_real, reader_real], 1)] * 6
    out = paired_difference(pooled, real, rng)
    assert abs(out["difference"] - 0.5) < 1e-9 and out["verdict"] == BETTER
    swapped = paired_difference([p[:, ::-1] for p in pooled], [r[:, ::-1] for r in real], rng)
    assert swapped["verdict"] == WORSE


def test_kappa_interval_matches_cohen_kappa_at_its_point():
    rng = np.random.default_rng(0)
    a, b = flags(50, 100), np.roll(flags(50, 100), 10)
    out = kappa_interval(a, b, rng)
    assert out["kappa"] == cohen_kappa(a, b) and out["interval"][0] < out["kappa"] < out["interval"][1] and out["n"] == 100


def test_paired_kappa_difference_is_zero_for_the_same_retest_and_positive_for_a_consistent_reader():
    rng = np.random.default_rng(0)
    first = flags(30, 60)
    noisy = np.roll(first, 15)
    same = paired_kappa_difference((first, noisy), (first, noisy), rng)
    assert same["difference"] == 0 and same["interval"] == [0.0, 0.0]
    better = paired_kappa_difference((first, noisy), (first, first), rng)
    assert better["difference"] > 0 and better["interval"][0] > 0


def test_kappa_ceiling_uses_both_retests_and_one_for_a_deterministic_reader():
    assert abs(kappa_ceiling(0.36, 0.64) - 0.48) < 1e-9 and abs(kappa_ceiling(0.49) - 0.7) < 1e-9
    assert kappa_ceiling(-0.1, 0.5) == 0.0
