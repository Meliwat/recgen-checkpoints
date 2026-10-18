import numpy as np

from recgen.analyze import adjacency_threshold, trace_neuron, window_top
from recgen.facets import TOP_K
from recgen.rubric import Facet


def test_window_top_returns_strongest_windows_with_their_positions():
    acts = np.zeros((3, 8))
    acts[0, 2], acts[1, 5], acts[2, 7] = 1.0, 3.0, 2.0
    assert window_top(acts, 2) == [(1, 5, 3.0), (2, 7, 2.0)]


def test_facet_response_is_relative_to_the_neurons_top_level_at_that_checkpoint():
    acts = np.zeros((TOP_K + 10, 8))
    acts[:, 5] = 2.0  # every window peaks at 2.0, so the top-50 level is 2.0
    acts[0, 6] = 1.0
    point = trace_neuron(acts, facets=[[(0, 5, 9.9), (1, 5, 9.9)], [(0, 6, 9.9), (1, 6, 9.9), (2, 6, 9.9)]])
    assert point["reference"] == 2.0
    assert point["responses"] == [1.0, 0.0]  # medians: 2.0 and 0.0


def test_adjacency_threshold_only_compares_facets_of_different_neurons():
    def f(vector):
        return Facet(np.ones(1), np.array(vector, dtype=float), "t")

    facets = {1: [f([1, 0]), f([1, 0])], 2: [f([0, 1])]}
    # Cross-neuron similarities are both 0; the same-neuron pair (similarity 1) is ignored.
    assert adjacency_threshold(facets) == 0.0
