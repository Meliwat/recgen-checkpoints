import numpy as np

from recgen.analyze_v13 import final_adjacency


def units(start, k=100):
    return np.arange(start, start + k)


def test_final_adjacency_uses_the_other_neurons_facets_as_the_null():
    # Neuron 1 has facets 0 and 1 sharing half their drivers; facet 2 shares nothing.
    # Neuron 2's two facets (read through neuron 1) share nothing with neuron 1's facets.
    candidates = [{"neuron": 1, "facets": [[], [], []]}, {"neuron": 2, "facets": [[], []]}]
    mlp = {(1, (1, 0)): units(0), (1, (1, 1)): np.concatenate([units(0, 50), units(1000, 50)]),
           (1, (1, 2)): units(5000), (1, (2, 0)): units(9000), (1, (2, 1)): units(9500)}
    overlaps, adjacent = final_adjacency({"mlp": mlp}, candidates, 1)
    assert overlaps == {"0-1": 0.5, "0-2": 0.0, "1-2": 0.0}
    assert adjacent["0-1"]["v2"] and not adjacent["0-2"]["v2"] and not adjacent["1-2"]["v2"]
