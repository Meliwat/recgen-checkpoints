import numpy as np

from recgen.rubric import (
    MIXED_FROM_START, ONE_ORIGIN_UNRESOLVED, PACKING_LIKE, RECURSIVE_GENERALIZATION, UNSTABLE,
    Facet, classify, emergence, tokens_related,
)

THRESHOLD = 0.5
# Three topics: A and B are related (cosine 0.8), C is unrelated to both.
A = np.array([1.0, 0.0, 0.0])
B = np.array([0.8, 0.6, 0.0])
C = np.array([0.0, 0.0, 1.0])


_unique_tokens = (f"token{i:06d}" for i in range(10**6))  # same length, so never substrings


def facet(responses, centroid, token=None):
    # Distinct tokens by default, so only the centroids decide adjacency.
    return Facet(np.array(responses, dtype=float), centroid, token or next(_unique_tokens))


def test_emergence_is_first_checkpoint_after_which_the_facet_stays_present():
    assert emergence(np.array([0.1, 0.6, 0.7, 0.9])) == 1


def test_emergence_tolerates_one_dip_but_not_two():
    assert emergence(np.array([0.6, 0.2, 0.7, 0.9])) == 0
    assert emergence(np.array([0.6, 0.2, 0.3, 0.9])) == 3


def test_facet_absent_at_the_end_never_emerges():
    assert emergence(np.array([0.9, 0.9, 0.4])) is None


def test_tokens_related_by_equality_or_substring_of_two_or_more_chars():
    assert tokens_related("iy", "diy")
    assert tokens_related(",", ",")
    assert not tokens_related("i", "diy")
    assert not tokens_related("moon", "cesium")


def test_later_facet_related_to_an_earlier_one_is_recursive_generalization():
    facets = [facet([0.1, 0.7, 0.9, 1.0], A), facet([0.1, 0.1, 0.6, 1.0], B)]
    assert classify(facets, THRESHOLD) == RECURSIVE_GENERALIZATION


def test_unrelated_facet_reached_through_a_bridge_is_recursive_generalization():
    # C is semantically unrelated to A and B, but shares B's activating token.
    facets = [facet([0.7, 0.9, 0.9, 1.0], A, "iy"),
              facet([0.1, 0.6, 0.9, 1.0], B, "diy"),
              facet([0.1, 0.1, 0.6, 1.0], C, "diy")]
    assert classify(facets, THRESHOLD) == RECURSIVE_GENERALIZATION


def test_later_facet_unrelated_to_everything_earlier_is_packing_like():
    facets = [facet([0.1, 0.7, 0.9, 1.0], A), facet([0.1, 0.1, 0.6, 1.0], C)]
    assert classify(facets, THRESHOLD) == PACKING_LIKE


def test_unrelated_facets_appearing_together_are_mixed_from_the_start():
    facets = [facet([0.1, 0.7, 0.9, 1.0], A), facet([0.1, 0.6, 0.9, 1.0], C)]
    assert classify(facets, THRESHOLD) == MIXED_FROM_START


def test_related_facets_appearing_together_have_unresolved_timing():
    facets = [facet([0.1, 0.7, 0.9, 1.0], A), facet([0.1, 0.6, 0.9, 1.0], B)]
    assert classify(facets, THRESHOLD) == ONE_ORIGIN_UNRESOLVED


def test_facet_that_never_settles_makes_the_neuron_unstable():
    facets = [facet([0.1, 0.7, 0.9, 1.0], A), facet([0.6, 0.1, 0.6, 0.1], B)]
    assert classify(facets, THRESHOLD) == UNSTABLE
