import numpy as np

from recgen.rubric_v5 import AGAINST, FITS, NO_TREND
from recgen.rubric_v11 import (
    ALL_DEPTHS, COMPARISONS, CORROBORATED, DEPTH_PAIRS, DEPTHS, NO_READER_TREND, READER_DEPENDENT, compare_depths,
    depth_key, excess_against_size, excess_over_models, reader_dependence, rng_for,
)


def flags(share: float, n: int) -> np.ndarray:
    return np.arange(n) < round(share * n)


def test_depths_cover_the_two_new_layers_and_compare_every_pair_with_the_middle():
    assert DEPTHS == ("quarter", "three quarters") and "middle" in ALL_DEPTHS and "middle" not in DEPTHS
    assert len(DEPTH_PAIRS) == 3 and len({frozenset(p) for p in DEPTH_PAIRS}) == 3 and COMPARISONS == 9
    assert depth_key("three quarters") == "three_quarters"


def test_excess_is_the_mean_over_models_of_pooled_minus_real():
    per_model = [(flags(0.7, 50), flags(0.4, 100)), (flags(0.5, 50), flags(0.4, 100))]
    out = excess_over_models(per_model, rng_for("quarter", 0))
    assert abs(out["excess"] - 0.2) < 1e-9 and len(out["draws"]) == 2000
    assert out["interval"][0] < 0.2 < out["interval"][1]


def test_excess_draws_are_reproducible_from_the_seed():
    per_model = [(flags(0.6, 50), flags(0.4, 100))]
    a = excess_over_models(per_model, rng_for("quarter", 1))
    b = excess_over_models(per_model, rng_for("quarter", 1))
    assert np.array_equal(a["draws"], b["draws"])


def test_a_large_drop_in_excess_between_depths_differs_and_survives_correction():
    strong = excess_over_models([(flags(0.8, 50), flags(0.3, 100))] * 6, rng_for("quarter", 2))
    weak = excess_over_models([(flags(0.4, 50), flags(0.3, 100))] * 6, rng_for("middle", 2))
    out = compare_depths(strong, weak)
    assert abs(out["difference"] - 0.4) < 1e-9 and out["differ"] and out["survives_correction"]
    assert out["corrected_interval"][0] < out["interval"][0] and out["corrected_interval"][1] > out["interval"][1]


def test_equal_excesses_do_not_differ():
    per_model = [(flags(0.6, 50), flags(0.4, 100))] * 6
    out = compare_depths(excess_over_models(per_model, rng_for("quarter", 3)),
                         excess_over_models(per_model, rng_for("middle", 3)))
    assert out["difference"] == 0 and not out["differ"] and not out["survives_correction"]


def test_v7s_middle_layer_pattern_is_reader_dependent():
    # MiniLM falls with size but can't see pooling; the two readers that can see no trend.
    verdicts = {"minilm": FITS, "network": NO_TREND, "llm": NO_TREND}
    intervals = {"minilm": [-0.01, 0.075], "network": [0.157, 0.282], "llm": [0.137, 0.273]}
    out = reader_dependence(verdicts, intervals)
    assert out["outcome"] == READER_DEPENDENT and out["sensitive"] == ["network", "llm"]
    assert out["corroborated_by"] == {"minilm": []}


def test_a_trend_seen_by_two_sensitive_readers_is_corroborated():
    verdicts = {"minilm": NO_TREND, "network": AGAINST, "llm": AGAINST}
    out = reader_dependence(verdicts, {"minilm": [0.0, 0.1], "network": [0.1, 0.3], "llm": [0.05, 0.2]})
    assert out["outcome"] == CORROBORATED and out["corroborated_by"] == {"network": ["llm"], "llm": ["network"]}


def test_corroboration_needs_a_sensitive_reader():
    # MiniLM agrees with the network reader, but MiniLM's excess interval reaches zero.
    verdicts = {"minilm": FITS, "network": FITS, "llm": NO_TREND}
    out = reader_dependence(verdicts, {"minilm": [-0.02, 0.1], "network": [0.1, 0.3], "llm": [0.05, 0.2]})
    assert out["outcome"] == READER_DEPENDENT and out["corroborated_by"] == {"minilm": ["network"], "network": []}


def test_opposite_trends_are_reader_dependent_and_no_trends_are_their_own_outcome():
    intervals = {"minilm": [0.1, 0.2], "network": [0.1, 0.3], "llm": [0.05, 0.2]}
    assert reader_dependence({"minilm": FITS, "network": AGAINST, "llm": NO_TREND}, intervals)["outcome"] == READER_DEPENDENT
    assert reader_dependence({r: NO_TREND for r in intervals}, intervals)["outcome"] == NO_READER_TREND


def test_excess_against_size_is_a_rank_correlation():
    assert excess_against_size([0.4, 0.3, 0.25, 0.2, 0.15, 0.1]) == -1.0
    assert excess_against_size([0.1, 0.2, 0.3, 0.35, 0.4, 0.5]) == 1.0
