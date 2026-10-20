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


def test_chance_tail_matches_a_binomial_when_rates_are_equal():
    from math import comb

    from recgen.analyze_v13 import chance_tail

    expected = sum(comb(20, k) * 0.1 ** k * 0.9 ** (20 - k) for k in range(4, 21))
    assert abs(chance_tail([0.1] * 20, 4) - expected) < 1e-12
    assert abs(chance_tail([0.1] * 20, 0) - 1) < 1e-12


def test_same_token_check_splits_pairs_by_shared_top_token():
    import numpy as np

    from recgen.analyze_v13 import same_token_check

    corpus = np.array([[7, 7, 8, 9]])
    candidates = [{"neuron": 1, "facets": [[[0, 0, 0.0], [0, 1, 0.0]], [[0, 1, 0.0]], [[0, 2, 0.0], [0, 3, 0.0], [0, 2, 0.0]]]}]
    pair = {"window": [128, 256], "v2_adjacent": False, "null_exceedances": 1, "null_size": 10}
    results = [{"neuron": 1, "mlp": {"pairs": {"0-1": {**pair, "branched_then_separated": True},
                                               "0-2": {**pair, "branched_then_separated": False},
                                               "1-2": {**pair, "branched_then_separated": False, "v2_adjacent": True}}}}]
    out = same_token_check(candidates, results, corpus)
    assert out["same top token"]["unrelated_pairs"] == 1 and out["same top token"]["branched_then_separated"] == 1
    assert out["different top tokens"]["unrelated_pairs"] == 1 and out["different top tokens"]["expected_by_chance"] == 0.1
