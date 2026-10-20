import numpy as np

from recgen.rubric_v12 import COVARIATES, h1, h2, neuron_stats, quartiles, top_minus_rest


def test_neuron_stats():
    window_max = np.concatenate([[10.0], np.linspace(5, 4, 49), np.ones(50)])  # 50th largest is 4
    s = neuron_stats(window_max, [7] * 30 + [8] * 20)
    assert s["peak"] == 10.0 and s["peakedness"] == 2.5 and s["token_purity"] == 0.6
    assert s["density"] == 2 / 100  # the peak and the 5.0 window reach half of 10


def test_density_counts_windows_at_or_above_half_the_peak():
    window_max = np.array([10.0, 5.0, 4.9, 1.0] + [0.5] * 46)
    assert neuron_stats(window_max, [1] * 50)["density"] == 2 / 50


def test_quartiles_split_four_ways():
    assert list(np.bincount(quartiles(np.arange(100.0)))) == [25, 25, 25, 25]


def test_top_minus_rest_and_h1_detect_a_negative_effect():
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(6):
        peak = rng.normal(size=400)
        flag = (rng.random(400) < np.where(quartiles(peak) == 3, 0.2, 0.5)).astype(float)
        groups.append((peak, quartiles(peak), flag))
    point, (low, high) = top_minus_rest([(q, f) for _, q, f in groups], rng)
    assert -0.4 < point < -0.2 and high < 0
    assert h1(groups, rng)["holds"]


def test_h2_recovers_a_peak_effect_that_survives_controls():
    rng = np.random.default_rng(1)
    rows = []
    for model in ("a", "b"):
        for _ in range(600):
            r = {c: rng.normal() for c in COVARIATES}
            p = 1 / (1 + np.exp(-(-1.0 * r["log_peak"] + 0.0 * r["token_purity"])))
            rows.append({**r, "model": model, "flag": float(rng.random() < p)})
    out = h2(rows, rng)
    assert out["holds"] and out["log_peak"]["coefficient"] < -0.5


def test_peakedness_is_undefined_below_50_firing_windows_and_h2_leaves_those_out():
    window_max = np.array([3.0] * 10 + [0.0] * 90)
    assert np.isnan(neuron_stats(window_max, [1] * 50)["peakedness"])
    rng = np.random.default_rng(2)
    rows = [{c: rng.normal() for c in COVARIATES} | {"model": "a", "flag": float(i % 2)} for i in range(200)]
    rows[0]["peakedness"] = float("nan")
    assert h2(rows, rng)["left_out_undefined_peakedness"] == 1
