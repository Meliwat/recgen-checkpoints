import numpy as np
import torch

from recgen.analyze_v3 import facet_rows, negative_control
from recgen.outputs import control_summary
from recgen.rubric_v3 import spearman_brown

V = 60


def specific(x, generic):
    unit = generic / np.linalg.norm(generic)
    return x - (x @ unit) * unit


def cos(x, y):
    return x @ y / (np.linalg.norm(x) * np.linalg.norm(y))


def test_facet_rows_keep_v1_examples_and_added_ones_above_the_cutoff():
    facets_v3 = [{"facets": [[[0, 40, 2.0, 0.9, True], [1, 50, 1.5, 0.45, False], [2, 60, 1.4, 0.55, False]]]},
                 {"facets": [[[3, 70, 2.0, 0.3, True]]]}]
    data = {"examples": np.array([[0, 0, 0, 40], [0, 0, 1, 50], [0, 0, 2, 60], [1, 0, 3, 70]])}
    assert facet_rows(data, facets_v3, 0, 0.4) == [[0, 1, 2]]
    assert facet_rows(data, facets_v3, 0, 0.5) == [[0, 2]]
    assert facet_rows(data, facets_v3, 1, 0.5) == [[3]]


def test_negative_control_matches_a_direct_computation():
    rng = np.random.default_rng(0)
    n = 12
    # Neuron 0 has facet A (rows 0-11); neuron 1 has facet X (rows 12-23).
    examples = np.array([[0, 0, w, 40] for w in range(n)] + [[1, 0, w, 40] for w in range(n, 2 * n)])
    effects = rng.standard_normal((2 * n, V)).astype(np.float16)
    generic = rng.standard_normal((2, 2, V)).astype(np.float32)
    # Neuron 0 patched into X's contexts: one effect per X example, split into even and odd halves.
    patched = rng.standard_normal((n, V)) + 2 * effects[:n].astype(np.float64).mean(axis=0)
    halves = np.stack([patched[0::2].mean(axis=0), patched[1::2].mean(axis=0)])
    control_sum = torch.zeros((2, 2, 2, V))
    control_sum[0, 1] = torch.from_numpy(halves * (n // 2)).float()
    count = np.array([[0, 0], [n // 2, n // 2]])
    data = {"examples": examples, "generic": generic, "control_count": count,
            **control_summary(effects, examples, control_sum, count, generic[:, 0])}
    facets_v3 = [{"neuron": 7, "facets": [[[w, 40, 2.0, 0.9, True] for w in range(n)]]},
                 {"neuron": 8, "facets": [[[w, 40, 2.0, 0.9, True] for w in range(n, 2 * n)]]}]
    results = [{"facets": [{"n": n, "reliability": 0.8}]}, {"facets": [{"n": n, "reliability": 0.0}]}]
    scores = negative_control(data, {"zero": effects}, facets_v3, results)

    g = generic[0, 0].astype(np.float64)
    a = specific(effects[:n].astype(np.float64).mean(axis=0), g)
    x = specific(halves.mean(axis=0), g)
    rel_x = spearman_brown(cos(specific(halves[0], g), specific(halves[1], g)))
    assert len(scores) == 1  # neuron 1's facet is unreliable, so only A against X counts
    np.testing.assert_allclose(scores[0], cos(a, x) / np.sqrt(0.8 * rel_x), rtol=1e-4)
