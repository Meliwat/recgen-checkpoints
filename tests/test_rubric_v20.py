import numpy as np

from recgen.rubric_v12 import quartiles
from recgen.rubric_v20 import (
    MIXED, MIXED_SIGN, NO, PERSISTS, RANKS, TOP50_SPECIFIC, UNCLEAR, YES, against_v17, answer, rank_rng, t2,
)


def test_rank_generators_are_per_neuron_and_per_band():
    assert rank_rng("70m", 5, "band").random() == rank_rng("70m", 5, "band").random()
    assert rank_rng("70m", 5, "band").random() != rank_rng("70m", 6, "band").random()
    assert rank_rng("70m", 5, "band").random() != rank_rng("70m", 5, "band2").random()
    assert rank_rng("70m", 5, "band").random() != rank_rng("160m", 5, "band").random()


def test_bands_are_disjoint_and_below_the_top_50():
    (a0, a1), (b0, b1) = RANKS["band"], RANKS["band2"]
    assert 50 < a0 <= a1 < b0 <= b1 <= 4096


def test_answer_needs_both_co_primary_verdicts():
    assert answer(PERSISTS, PERSISTS) == YES
    assert answer(TOP50_SPECIFIC, TOP50_SPECIFIC) == NO
    assert answer(PERSISTS, TOP50_SPECIFIC) == MIXED and answer(PERSISTS, UNCLEAR) == MIXED


def test_against_v17_is_v17s_t2_with_v17s_band_in_place_of_the_top_50():
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(6):
        peak = rng.normal(size=200)
        q = quartiles(peak)
        groups.append((peak, q, (rng.random(200) < 0.4).astype(float), (rng.random(200) < np.where(q == 3, 0.2, 0.5)).astype(float)))
    out = against_v17(groups, MIXED_SIGN["flag"], np.random.default_rng(1))
    expected = t2(groups, MIXED_SIGN["flag"], np.random.default_rng(1))
    assert out["rank_band"] == expected["band"] and out["v17_band"] == expected["top"]
    assert out["rank_minus_v17"] == expected["band_minus_top"]
