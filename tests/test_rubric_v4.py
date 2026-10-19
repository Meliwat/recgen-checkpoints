import numpy as np

from recgen.rubric_v4 import (
    adjacent_at_formation, branched_then_separated, crossing_at, formation_threshold, formation_window,
    pair_window, peak,
)


def test_crossing_is_first_checkpoint_after_which_the_gain_stays_above_the_fraction():
    gains = np.array([0.0, 0.3, 0.6, 0.8, 1.0])
    assert crossing_at(gains, 0.25) == 1
    assert crossing_at(gains, 0.75) == 3


def test_crossing_tolerates_one_dip_and_needs_a_positive_final_gain():
    assert crossing_at(np.array([0.5, 0.1, 0.6, 1.0]), 0.25) == 0
    assert crossing_at(np.array([0.5, 0.1, 0.1, 1.0]), 0.25) == 3
    assert crossing_at(np.array([0.5, 0.8, -0.1]), 0.25) is None


def test_formation_window_runs_from_quarter_to_three_quarters_of_the_final_gain():
    assert formation_window(np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])) == (2, 4)


def test_a_facet_that_jumps_at_once_has_a_one_checkpoint_window():
    assert formation_window(np.array([0.0, 0.0, 0.9, 1.0])) == (2, 2)


def test_pair_window_spans_both_facets_and_needs_both():
    assert pair_window((2, 4), (5, 6)) == (2, 6)
    assert pair_window((2, 4), None) is None


def test_peak_only_looks_inside_the_window():
    assert peak(np.array([0.9, 0.1, 0.2, 0.9]), (1, 2)) == 0.2


def test_adjacent_when_the_pair_peaks_above_the_null_peaks_in_the_same_window():
    window = (1, 2)
    null = [np.array([0.9, 0.02, 0.03, 0.9])] * 19 + [np.array([0.0, 0.05, 0.04, 0.0])]
    # The null's in-window peaks are 0.03 (19 times) and 0.05; its 95th percentile is 0.031.
    # Out-of-window values (0.9) must not count.
    assert formation_threshold(null, window) < 0.05
    assert adjacent_at_formation(np.array([0.0, 0.1, 0.0, 0.0]), null, window)
    assert not adjacent_at_formation(np.array([0.9, 0.02, 0.0, 0.9]), null, window)


def test_branched_then_separated_means_shared_at_formation_but_not_at_the_end():
    assert branched_then_separated(True, False)
    assert not branched_then_separated(True, True)
    assert not branched_then_separated(False, False)
