import numpy as np
import pytest

from recgen import rubric
from recgen.rubric_v2 import (
    BRIDGE_LEADS, BRIDGE_WITH_OR_AFTER, NO_BRIDGE_RESPONSE, WINDOW_TOO_SHORT,
    adjacency, bridge_outcome, classify, crossing, dense_grid, drivers, null_threshold, overlap,
)


def test_drivers_are_the_largest_positive_contributions():
    contributions = np.array([0.5, -2.0, 3.0, 0.0, 1.0])
    assert drivers(contributions, k=2).tolist() == [2, 4]
    assert drivers(contributions, k=4).tolist() == [2, 4, 0]  # 0.0 and -2.0 never drive


def test_overlap_is_a_fraction_of_k_even_when_fewer_drivers_exist():
    assert overlap(np.array([1, 2, 3]), np.array([2, 3, 9]), k=4) == 0.5


def test_adjacent_only_when_overlap_exceeds_the_null_threshold():
    overlaps = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.1], [0.3, 0.1, 1.0]])
    threshold = null_threshold([0.0, 0.1, 0.2, 0.2, 0.2], percentile=50)  # 0.2
    assert adjacency(overlaps, threshold).tolist() == [[True, False, True], [False, True, False], [True, False, True]]


# v1's test cases: A and B are related (cosine 0.8), C is unrelated to both.
A, B, C = np.array([1.0, 0.0, 0.0]), np.array([0.8, 0.6, 0.0]), np.array([0.0, 0.0, 1.0])
V1_CASES = [
    [([0.1, 0.7, 0.9, 1.0], A, "a"), ([0.1, 0.1, 0.6, 1.0], B, "b")],
    [([0.7, 0.9, 0.9, 1.0], A, "iy"), ([0.1, 0.6, 0.9, 1.0], B, "diy"), ([0.1, 0.1, 0.6, 1.0], C, "diy")],
    [([0.1, 0.7, 0.9, 1.0], A, "a"), ([0.1, 0.1, 0.6, 1.0], C, "c")],
    [([0.1, 0.7, 0.9, 1.0], A, "a"), ([0.1, 0.6, 0.9, 1.0], C, "c")],
    [([0.1, 0.7, 0.9, 1.0], A, "a"), ([0.1, 0.6, 0.9, 1.0], B, "b")],
    [([0.1, 0.7, 0.9, 1.0], A, "a"), ([0.6, 0.1, 0.6, 0.1], B, "b")],
]


@pytest.mark.parametrize("case", V1_CASES)
def test_classify_matches_v1_when_given_v1_adjacency(case):
    facets = [rubric.Facet(np.array(r), c, t) for r, c, t in case]
    adjacent = np.array([[rubric.adjacent(f, g, 0.5) for g in facets] for f in facets])
    times = [rubric.emergence(f.responses) for f in facets]
    assert classify(times, adjacent) == rubric.classify(facets, 0.5)


def test_two_facet_neuron_can_now_be_recursive_generalization():
    adjacent = np.array([[True, True], [True, True]])
    assert classify([1, 2], adjacent) == rubric.RECURSIVE_GENERALIZATION
    assert classify([1, 2], ~adjacent) == rubric.PACKING_LIKE


def test_dense_grid_adds_evenly_spaced_thousands_between_v1_checkpoints():
    grid = [0, 512, 1000, 2000, 3000, 5000, 8000, 13000]
    assert dense_grid(grid, per_interval=3) == [0, 512, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
                                                9000, 10000, 12000, 13000]


def test_crossing_is_where_the_gain_stays_above_half_its_last_value():
    assert crossing(np.array([0.0, 0.2, 0.6, 0.8, 1.0])) == 2
    assert crossing(np.array([0.0, 0.6, 0.3, 0.7, 1.0])) == 1  # one dip allowed
    assert crossing(np.array([0.0, 0.6, 0.3, 0.2, 1.0])) == 4  # two are not
    assert crossing(np.array([0.5, 0.2, -0.1])) is None


FACET = np.array([0.0, 0.1, 0.2, 0.5, 1.0])  # crosses half at index 3


def test_bridge_that_rises_two_checkpoints_before_the_facet_leads():
    assert bridge_outcome(FACET, np.array([0.0, 0.6, 0.7, 0.8, 0.8])) == BRIDGE_LEADS


def test_bridge_that_already_responds_when_the_window_opens_leads():
    assert bridge_outcome(FACET, np.array([0.7, 0.8, 0.7, 0.8, 0.8])) == BRIDGE_LEADS


def test_bridge_rising_in_lockstep_with_the_facet_does_not_lead():
    assert bridge_outcome(FACET, 0.3 * FACET) == BRIDGE_WITH_OR_AFTER


def test_bridge_barely_responding_by_the_facets_emergence_is_no_bridge():
    assert bridge_outcome(FACET, np.array([0.0, 0.1, 0.1, 0.1, 0.1])) == NO_BRIDGE_RESPONSE


def test_two_checkpoint_window_cannot_show_a_lead():
    assert bridge_outcome(np.array([0.0, 1.0]), np.array([0.8, 0.8])) == WINDOW_TOO_SHORT
