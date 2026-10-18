import numpy as np

from recgen.facets import find_facets, top_examples


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def group(direction, n, rng):
    return unit(np.array(direction, dtype=float) + 0.05 * rng.normal(size=(n, len(direction))))


def test_separated_groups_become_facets_largest_first_and_strays_are_dropped():
    rng = np.random.default_rng(0)
    embeddings = np.vstack([
        group([0, 1, 0, 0], 7, rng),
        group([1, 0, 0, 0], 10, rng),
        unit(np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)),
    ])
    facets = find_facets(embeddings)
    assert [sorted(f.tolist()) for f in facets] == [list(range(7, 17)), list(range(7))]


def test_one_tight_group_is_one_facet():
    facets = find_facets(group([1, 0, 0], 12, np.random.default_rng(1)))
    assert len(facets) == 1 and len(facets[0]) == 12


def test_groups_smaller_than_the_minimum_are_not_facets():
    rng = np.random.default_rng(2)
    embeddings = np.vstack([group([1, 0, 0], 8, rng), group([0, 1, 0], 4, rng)])
    assert [len(f) for f in find_facets(embeddings)] == [8]


def test_top_examples_are_one_per_window_strongest_first():
    window_max = np.array([[0.1, 5.0], [0.9, 1.0], [0.5, 2.0]])
    window_pos = np.array([[40, 41], [50, 51], [60, 61]], dtype=np.int16)
    assert top_examples(window_max, window_pos, neuron=0, k=2) == [(1, 50, 0.9), (2, 60, 0.5)]
