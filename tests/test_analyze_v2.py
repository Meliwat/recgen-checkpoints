import numpy as np

from recgen.analyze_v2 import driver_overlaps, later_facets_with_bridge
from recgen.rubric_v2 import BRIDGE_LEADS, BRIDGE_WITH_OR_AFTER, WINDOW_TOO_SHORT


def test_null_reads_other_neurons_facets_through_this_neurons_coefficients():
    # 4 upstream units. Neuron 7 is driven by units 0 and 1, neuron 8 by units 2 and 3.
    coefficients = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    att = {"coefficients": coefficients, "neurons": [7, 8]}
    shifts = {(7, 0): np.array([1.0, 0.5, 0.0, 0.0]),
              (7, 1): np.array([0.2, 1.0, 0.0, 0.0]),
              (8, 0): np.array([0.0, 0.0, 1.0, 1.0]),
              (8, 1): np.array([1.0, 0.0, 1.0, 0.0])}
    pairs, null = driver_overlaps(att, shifts, neuron=7, k=2)
    assert pairs.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    # Read through neuron 7, facet (8, 0) has no drivers and facet (8, 1) has unit 0 only.
    assert null == [0.0, 0.5, 0.0, 0.5]


def test_later_facet_counts_once_however_many_earlier_facets_bridge_to_it():
    bridges = [{"neuron": 1, "b": 2, "bridge": BRIDGE_LEADS},
               {"neuron": 1, "b": 2, "bridge": BRIDGE_LEADS},
               {"neuron": 1, "b": 3, "bridge": BRIDGE_WITH_OR_AFTER},
               {"neuron": 5, "b": 0, "bridge": WINDOW_TOO_SHORT}]
    assert later_facets_with_bridge(bridges) == {"testable": 2, "with a leading bridge": 1}


def test_chance_labels_for_a_staggered_pair_follow_the_false_positive_rate():
    from recgen.analyze_v2 import chance_labels
    from recgen.rubric import PACKING_LIKE, RECURSIVE_GENERALIZATION
    labels = chance_labels([3, 5], rate=0.05)
    assert labels[RECURSIVE_GENERALIZATION] == 0.05
    assert labels[PACKING_LIKE] == 0.95
