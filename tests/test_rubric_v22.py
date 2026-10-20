import numpy as np
import pytest

from recgen.review_v19 import SIZE_MATCHED
from recgen.rubric_v22 import (
    CORPUS_MOST, CORPUS_PART, NOT_CORPUS, PYTHIA_SIZE_MATCHED, decide_c, excess_change, rate_change, rng,
)


def flags(n_true: int, n: int) -> np.ndarray:
    return np.r_[np.ones(n_true), np.zeros(n - n_true)]


def test_the_pythia_target_uses_v19s_size_matched_sizes():
    assert PYTHIA_SIZE_MATCHED == tuple(SIZE_MATCHED.values())


def test_rate_change_is_the_mean_paired_difference_over_models():
    owt, pile = flags(30, 100), flags(10, 100)  # +20 points, the same units flagged in both and 20 more
    out = rate_change([(owt, pile)] * 3, rng(0))
    assert abs(out["change"] - 0.2) < 1e-12 and 0.1 < out["interval"][0] < 0.2 < out["interval"][1]
    unchanged = rate_change([(owt, owt)] * 3, rng(0))
    assert unchanged["change"] == 0 and unchanged["interval"] == [0, 0]  # paired: no change, no spread


def test_rate_change_needs_flags_paired_by_unit():
    with pytest.raises(AssertionError):
        rate_change([(flags(3, 10), flags(3, 9))], rng(0))


def test_excess_change_is_the_change_in_pools_minus_the_change_in_real_units():
    pooled = (flags(30, 50), flags(20, 50))  # +20 points on pools
    real = (flags(40, 100), flags(10, 100))  # +30 points on real units
    out = excess_change([(pooled, real)] * 3, rng(1))
    assert abs(out["change"] + 0.1) < 1e-12 and out["interval"][0] < -0.1 < out["interval"][1]


def test_c_reads_the_rise_against_the_gap_to_pythia():
    assert decide_c(0.32, 0.17, 0.42, [0.08, 0.22])["verdict"] == CORPUS_MOST  # 60% of the gap
    out = decide_c(0.25, 0.17, 0.42, [0.02, 0.14])
    assert out["verdict"] == CORPUS_PART and abs(out["share_of_gap_closed"] - 0.32) < 1e-12
    assert decide_c(0.35, 0.17, 0.42, [-0.01, 0.3])["verdict"] == NOT_CORPUS  # interval includes 0
    assert decide_c(0.10, 0.17, 0.42, [-0.12, -0.02])["verdict"] == NOT_CORPUS
    with pytest.raises(AssertionError):
        decide_c(0.5, 0.45, 0.42, [0.01, 0.1])
