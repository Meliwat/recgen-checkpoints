import numpy as np

from recgen.rubric_v12 import COVARIATES, quartiles
from recgen.rubric_v17 import (
    LESS_MIXED, MIXED_SIGN, MORE_MIXED, NO_DIFFERENCE, PERSISTS, TOP50_SPECIFIC, UNCLEAR, band_rng, ols_coefficients,
    reference_rng, t1, t2, verdict,
)


def test_band_and_reference_generators_are_per_neuron_and_distinct():
    assert band_rng("70m", 5).random() == band_rng("70m", 5).random()
    assert band_rng("70m", 5).random() != band_rng("70m", 6).random()
    assert band_rng("70m", 5).random() != reference_rng("70m", 5).random()


def test_t1_averages_models_equally_and_reads_the_mixed_side():
    rng = np.random.default_rng(0)
    pairs = [(np.ones(400), np.zeros(400)), (np.zeros(100), np.zeros(100))]  # +1 in one model, 0 in the other
    out = t1(pairs, MIXED_SIGN["flag"], rng)
    assert out["band_minus_top"] == 0.5 and out["by_model"] == [1.0, 0.0] and out["reading"] == MORE_MIXED
    assert t1(pairs, MIXED_SIGN["coherence"], rng)["reading"] == LESS_MIXED  # higher coherence is less mixed
    noise = [(rng.random(300), rng.random(300)) for _ in range(6)]
    assert t1(noise, 1, rng)["reading"] == NO_DIFFERENCE


def test_t2_paired_difference_is_band_statistic_minus_top_statistic():
    rng = np.random.default_rng(1)
    groups = []
    for _ in range(6):
        peak = rng.normal(size=300)
        q = quartiles(peak)
        top = (rng.random(300) < np.where(q == 3, 0.2, 0.5)).astype(float)  # a top-50 magnitude effect...
        band = (rng.random(300) < 0.6).astype(float)  # ...that's gone in the band
        groups.append((peak, q, band, top))
    out = t2(groups, MIXED_SIGN["flag"], rng)
    expected = out["band"]["top_minus_rest"] - out["top"]["top_minus_rest"]
    assert abs(out["band_minus_top"]["difference"] - expected) < 1e-12
    assert out["top"]["holds"] and not out["band"]["holds"]
    assert out["band_minus_top"]["interval"][0] > 0 and out["verdict"] == TOP50_SPECIFIC


def test_t2_on_coherence_holds_when_strong_neurons_are_more_coherent():
    rng = np.random.default_rng(2)
    groups = []
    for _ in range(6):
        peak = rng.normal(size=300)
        coherence = 60 + 10 * peak + rng.normal(0, 5, 300)
        groups.append((peak, quartiles(peak), coherence, coherence + 20))
    out = t2(groups, MIXED_SIGN["coherence"], rng)
    assert out["band"]["holds"] and out["band"]["top_minus_rest"] < 0 and out["verdict"] == PERSISTS


def test_verdict_rule():
    holds = {"holds": True, "interval": [-0.3, -0.1]}
    flat = {"holds": False, "interval": [-0.05, 0.04]}
    below_but_few_models = {"holds": False, "interval": [-0.3, -0.1]}
    assert verdict(holds, [0.0, 0.2]) == PERSISTS
    assert verdict(flat, [0.1, 0.3]) == TOP50_SPECIFIC
    assert verdict(flat, [-0.05, 0.2]) == UNCLEAR
    assert verdict(below_but_few_models, [0.1, 0.3]) == UNCLEAR


def test_ols_coefficients_recover_a_log_peak_slope():
    rng = np.random.default_rng(3)
    rows = []
    for model in ("a", "b"):
        for _ in range(500):
            r = {c: rng.normal() for c in COVARIATES}
            rows.append({**r, "model": model, "flag": -2.0 * r["log_peak"] + 0.5 * r["density"] + rng.normal()})
    out = ols_coefficients(rows, rng)
    assert out["holds"] and abs(out["log_peak"]["coefficient"] + 2.0) < 0.2
    assert out["density"]["interval"][0] > 0
