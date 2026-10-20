import numpy as np

from recgen.rubric_v7 import (
    MIXES, N_POOLED, N_REAL, has_power, is_specific, parse_label, pool_rows, prompt, random_pairs, real_units,
    related_pairs,
)


def test_real_units_are_seeded_and_capped_at_the_sample():
    a, b = real_units(2048, "1b"), real_units(2048, "1b")
    assert len(a) == N_REAL and np.array_equal(a, b) and not np.array_equal(a, real_units(2048, "2.8b"))
    assert len(real_units(60, "70m")) == 60


def test_random_pairs_differ_by_mix_and_never_pair_a_neuron_with_itself():
    pairs = {mix: random_pairs(500, "410m", mix) for mix in MIXES}
    assert all(len(p) == N_POOLED and all(a != b for a, b in p) for p in pairs.values())
    assert pairs["50/50"] != pairs["70/30"]


def test_related_pairs_are_the_most_alike_directions_without_reuse():
    directions = np.array([[1.0, 0.0], [0.99, 0.1], [0.98, 0.15], [0.0, 1.0], [0.1, 0.99]])
    # Most alike: 1-2 (0.999), then 3-4 and 0-1 (0.995); 0-1 is skipped because 1 is used.
    assert related_pairs(directions, n=2) == [(1, 2), (3, 4)]
    pairs = related_pairs(directions, n=3)
    used = [x for p in pairs for x in p]
    assert len(used) == len(set(used))


def test_pool_rows_takes_the_mix_and_skips_duplicates():
    a, b = np.arange(0, 50), np.array([3, 4] + list(range(100, 148)))
    pooled = pool_rows(a, b, from_a=45)
    assert len(pooled) == 50 and list(pooled[:45]) == list(range(45))
    assert list(pooled[45:]) == [100, 101, 102, 103, 104]  # 3 and 4 were already in a's 45


def test_prompt_numbers_snippets_and_asks_for_json():
    text = prompt(["a [[b]] c", "d [[e]] f"])
    assert "2 text snippets" in text and "1. a [[b]] c" in text and "2. d [[e]] f" in text and '"label"' in text


def test_parse_label_reads_json_and_rejects_anything_else():
    assert parse_label('{"label": "C", "patterns": ["x", "y"]}') == "C"
    assert parse_label('Sure.\n```json\n{"label": "A", "patterns": ["x"]}\n```') == "A"
    assert parse_label('{"label": "D"}') is None
    assert parse_label("no json here") is None


def test_power_needs_a_large_gap_and_separated_intervals():
    rng = np.random.default_rng(0)
    real = np.array([True] * 10 + [False] * 90)
    assert has_power(np.array([True] * 60 + [False] * 40), real, rng)[0]
    assert not has_power(np.array([True] * 25 + [False] * 75), real, rng)[0]


def test_specificity_compares_related_pools_with_random_pools():
    specific, gap = is_specific(np.array([True] * 2 + [False] * 8), np.array([True] * 7 + [False] * 3))
    assert specific and abs(gap - 0.5) < 1e-9
    assert not is_specific(np.array([True] * 6 + [False] * 4), np.array([True] * 7 + [False] * 3))[0]
