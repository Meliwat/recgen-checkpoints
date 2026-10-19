import numpy as np
import pytest

from recgen.activations import MIN_CONTEXT
from recgen.config import CONTEXT_LEN
from recgen.rubric_v6 import (
    MERGE_PERCENTILE, MINILM_PERCENTILES, N_RANDOM, SENSITIVITY_SETTINGS, UNRELATED_PERCENTILE, cohen_kappa, embed,
    find_facets, least_related, pair_similarities, partition, percentile_rank, random_positions, thresholds,
    token_purity, unrelated_mixed,
)


def test_random_positions_are_seeded_one_per_window_with_full_context():
    keys = random_positions(4096)
    assert keys.shape == (N_RANDOM, 2) and np.array_equal(keys, random_positions(4096))
    assert len(np.unique(keys[:, 0])) == N_RANDOM and np.all(np.diff(keys[:, 0]) > 0)
    assert keys[:, 1].min() >= MIN_CONTEXT and keys[:, 1].max() < CONTEXT_LEN


def test_sensitivity_settings_map_v5_distances_through_minilm_percentiles():
    for (distance, min_size), (percentile, size) in SENSITIVITY_SETTINGS.items():
        assert size == min_size and percentile == MINILM_PERCENTILES[round(1 - distance, 2)]
    assert (MERGE_PERCENTILE, UNRELATED_PERCENTILE) == (MINILM_PERCENTILES[0.25], MINILM_PERCENTILES[0.1])


def test_embed_centers_scales_projects_and_normalizes():
    rng = np.random.default_rng(0)
    z, mean, scale = rng.normal(size=(20, 8)), rng.normal(size=8), rng.uniform(0.5, 2, size=8)
    directions = rng.normal(size=(2, 8))
    units = embed(z, mean, directions, scale)
    np.testing.assert_allclose(np.linalg.norm(units, axis=1), 1)
    np.testing.assert_allclose(units @ directions.T, 0, atol=1e-12)
    expected = (z - mean) * scale
    basis = np.linalg.qr(directions.T)[0]
    expected -= expected @ basis @ basis.T
    np.testing.assert_allclose(units, expected / np.linalg.norm(expected, axis=1, keepdims=True))
    np.testing.assert_allclose(embed(z, None, None), z / np.linalg.norm(z, axis=1, keepdims=True))


def test_projection_stops_a_neurons_own_direction_from_grouping_its_examples():
    rng = np.random.default_rng(1)
    direction = rng.normal(size=64)
    random = rng.normal(size=(300, 64))
    examples = rng.normal(size=(50, 64)) + 4 * direction / np.linalg.norm(direction) * np.sqrt(64)
    mean = random.mean(axis=0)
    merge, _ = thresholds(pair_similarities(embed(random, mean, None)))
    assert len(find_facets(embed(examples, mean, None), merge)) == 1  # all alike by construction
    merge, _ = thresholds(pair_similarities(embed(random, mean, direction)))
    assert find_facets(embed(examples, mean, direction), merge) == []  # nothing else in common


def test_find_facets_merges_above_the_level_and_keeps_clusters_of_five():
    rng = np.random.default_rng(2)
    a = np.array([1.0, 0, 0, 0]) + 0.1 * rng.normal(size=(30, 4))
    b = np.array([0, 1.0, 0, 0]) + 0.1 * rng.normal(size=(15, 4))
    c = np.array([0, 0, 1.0, 0]) + 0.1 * rng.normal(size=(4, 4))
    units = embed(np.vstack([a, b, c]), None, None)
    facets = find_facets(units, merge_similarity=0.5)
    assert [len(f) for f in facets] == [30, 15]
    assert [len(f) for f in find_facets(units, merge_similarity=-0.9)] == [49]
    assert len(find_facets(units, merge_similarity=0.5, min_size=4)) == 3
    similarity, pair = least_related(units, facets)
    assert abs(similarity) < 0.1 and pair == (0, 1)
    assert least_related(units, facets[:1]) == (None, None)


def test_unrelated_mixed_uses_the_neurons_own_cutoff():
    assert unrelated_mixed(2, 0.05, 0.1) and not unrelated_mixed(2, 0.05, 0.02)
    assert not unrelated_mixed(1, None, 0.1)


def test_thresholds_and_percentile_rank_read_the_null():
    null = np.arange(1000) / 1000
    merge, unrelated = thresholds(null)
    assert merge == pytest.approx(np.percentile(null, MERGE_PERCENTILE))
    assert unrelated == pytest.approx(np.percentile(null, UNRELATED_PERCENTILE))
    assert percentile_rank(0.25, null) == 25.0


def test_partition_gives_examples_outside_facets_labels_of_their_own():
    labels = partition([np.array([0, 2]), np.array([3])], 5)
    assert labels[0] == labels[2] == 0 and labels[3] == 1
    assert len(set(labels)) == 4


def test_token_purity_and_cohen_kappa():
    assert token_purity([5, 5, 5, 7]) == 0.75
    flags = np.array([True, False] * 50)
    assert cohen_kappa(flags, flags) == 1.0
    assert cohen_kappa(flags, ~flags) == -1.0
    assert abs(cohen_kappa(flags, np.array([True] * 50 + [False] * 50))) < 1e-12
