import numpy as np

from recgen.analyze_v13 import final_adjacency
from recgen.analyze_v23 import neuron_rows, pre_registered


def units(start, k=100):
    return np.arange(start, start + k)


def test_final_adjacency_reads_head_drivers_as_top_5():
    candidates = [{"neuron": 1, "facets": [[], []]}, {"neuron": 2, "facets": [[]]}]
    heads = {(1, (1, 0)): np.arange(5), (1, (1, 1)): np.array([0, 1, 2, 10, 11]), (1, (2, 0)): np.arange(20, 25)}
    overlaps, adjacent, threshold = final_adjacency({"heads": heads}, candidates, 1, "heads")
    assert overlaps == {"0-1": 0.6} and threshold == 0.0 and adjacent["0-1"]["v2"]


def candidate(neuron, quartile, n_facets=2):
    return {"neuron": neuron, "quartile": quartile, "peak": 1.0 + quartile, "log_peak": 0.0, "unrelated_mixed": True,
            "reader_least_pair": "0-1", "facets": [[[0, 0, 0.0]] * 5] * n_facets}


def test_neuron_rows_measure_each_neuron_against_the_others_facets():
    # Neuron 1's facets share half their drivers; neuron 2's share none. Read through either neuron's
    # weights, a facet shares nothing with the other neuron's facets, so both thresholds are 0.
    candidates = [candidate(1, 3), candidate(2, 0)]
    mlp, heads = {}, {}
    for n in (1, 2):
        for m, start in ((1, 0), (2, 5000)):
            mlp[n, (m, 0)] = units(start)
            mlp[n, (m, 1)] = np.concatenate([units(start, 50), units(start + 1000, 50)]) if m == 1 else units(start + 2000)
            heads[n, (m, 0)] = np.arange(5) + 10 * m
            heads[n, (m, 1)] = np.arange(5) + 10 * m + 5
    rows = neuron_rows(candidates, {"mlp": mlp, "heads": heads})
    assert rows[0]["mlp"]["overlaps"] == {"0-1": 0.5} and not rows[0]["mlp"]["least_unrelated"]
    assert rows[1]["mlp"]["overlaps"] == {"0-1": 0.0} and rows[1]["mlp"]["least_unrelated"]
    assert rows[0]["mlp"]["threshold"] == 0.0 and rows[0]["mlp"]["margin"] == 0.5
    assert rows[0]["heads"]["least_unrelated"] and rows[0]["median_facet_size"] == 5.0


def test_pre_registered_reads_a_clear_gradient_as_more_jobs():
    # Strong neurons' least pairs share drivers (margin above 0); weak neurons' don't.
    rows = []
    for i in range(80):
        q = i % 4
        margin = 0.05 + 0.01 * (i % 3) if q == 3 else -0.05 - 0.01 * (i % 3)
        measures = {"least_unrelated": margin <= 0, "share_unrelated": float(margin <= 0), "margin": margin,
                    "reader_least_unrelated": margin <= 0}
        rows.append({"quartile": q, "peak": float(q) + i / 1000, "unrelated_mixed": margin <= 0, "n_facets": 2 + i % 3,
                     "median_facet_size": 5.0 + i % 4, "mlp": measures, "heads": measures})
    out = pre_registered(rows)
    assert out["mlp"]["P1_least_pair_unrelated"]["value"] == -1.0
    assert out["mlp"]["reading"] == "more jobs"
    assert out["controls"]["C2_least_pair_within_facet_count_strata"]["reading"] == "holds within strata"
