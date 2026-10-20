import numpy as np

import recgen.analyze_v20 as analyze_v20
from recgen.analyze_v20 import band_tests, by_quartile, comparison, in_v17_band, selectivity, v17_band_ranks
from recgen.rubric_v5 import MODELS
from recgen.rubric_v20 import PERSISTS, READINGS, TOP50_SPECIFIC


def synthetic_rows(rng) -> list[dict]:
    """Top-50 flags with a magnitude effect (strongest quartile 0.2, the rest 0.5); the rank band keeps it (0.3
    against 0.6); the second band and v17's band lose it (0.6 everywhere). The second band is missing for every
    tenth neuron."""
    rows = []
    for model in MODELS:
        for i in range(400):
            q = i % 4
            rows.append({"model": model, "quartile": q, "log_peak": q + rng.normal(0, 0.1),
                         "top_flag": float(rng.random() < (0.2 if q == 3 else 0.5)),
                         "band_flag": float(rng.random() < (0.3 if q == 3 else 0.6)),
                         "band2_flag": None if i % 10 == 0 else float(rng.random() < 0.6),
                         "v17_flag": float(rng.random() < 0.6)})
    return rows


def test_band_tests_give_each_band_its_verdict_and_reading():
    rows = synthetic_rows(np.random.default_rng(0))
    band = band_tests(rows, "flag", "band", 0)
    assert band["n"] == len(rows) and band["t2"]["verdict"] == PERSISTS and band["reading"] == READINGS[PERSISTS]
    second = band_tests(rows, "flag", "band2", 10)
    assert second["n"] == len(rows) - len(rows) // 10
    assert second["t2"]["verdict"] == TOP50_SPECIFIC and second["reading"] == "v17's persistence explained by selectivity"


def test_comparison_pairs_the_rank_band_with_v17s_band():
    rows = synthetic_rows(np.random.default_rng(1))
    rows[0]["v17_flag"] = None
    out = comparison(rows, "flag", 0)
    assert out["n"] == len(rows) - 1
    expected = out["rank_band"]["top_minus_rest"] - out["v17_band"]["top_minus_rest"]
    assert abs(out["rank_minus_v17"]["difference"] - expected) < 1e-12 and out["rank_minus_v17"]["interval"][1] < 0


def test_by_quartile_counts_each_sample_separately():
    rows = synthetic_rows(np.random.default_rng(2))
    out = by_quartile(rows, "flag")
    assert set(out) == {"top", "band", "band2", "v17"} and out["band2"]["n"] == len(rows) - len(rows) // 10
    assert out["top"]["mean"][3] < out["top"]["mean"][0]


def test_in_v17_band_includes_the_low_edge_only():
    assert in_v17_band(np.array([0.39, 0.4, 0.59, 0.6])).tolist() == [False, True, True, False]


def test_selectivity_pools_examples_within_model_and_quartile():
    rows = []
    for k, model in enumerate(MODELS):
        for q in range(4):
            fractions = np.full(50, 0.45 if q == 3 else 0.65) + 0.01 * k
            rows.append({"model": model, "quartile": q, "band_fractions": fractions, "band2_fractions": fractions - 0.2,
                         "v17_ranks": np.full(50, 100 * (q + 1))})
    out = selectivity(rows)
    assert out["band"]["median_fraction_of_peak"][0] == [0.65, 0.7] and out["band"]["median_fraction_of_peak"][3] == [0.45, 0.5]
    assert out["band"]["share_in_v17_band"][0] == [0.0, 0.0] and out["band"]["share_in_v17_band"][3] == [1.0, 1.0]
    assert out["band2"]["share_in_v17_band"][0] == [1.0, 1.0] and out["band2"]["share_in_v17_band"][3] == [0.0, 0.0]
    assert out["v17_band"]["median_rank"][1] == [200, 200]


def test_v17_band_ranks_rank_v17s_windows_with_v20s_ranking(tmp_path, monkeypatch):
    window_max = np.array([[3.0, 1.0], [5.0, 2.0], [1.0, 2.0], [4.0, 0.5]], dtype=np.float16)
    np.savez(tmp_path / "v5.npz", max=window_max)
    keys = np.array([[[0, 40], [2, 40]], [[-1, -1], [-1, -1]]])
    monkeypatch.setattr(analyze_v20, "sweep_path", lambda model, depth: tmp_path / "v5.npz")
    monkeypatch.setattr(analyze_v20, "band_examples", lambda model: {"neurons": np.array([0, 1]), "keys": keys,
                                                                      "included": np.array([True, False])})
    assert v17_band_ranks("70m").tolist() == [[3, 4], [0, 0]]
