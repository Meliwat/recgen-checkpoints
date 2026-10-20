import numpy as np

from recgen.analyze_v12 import reader_tests
from recgen.rubric_v5 import MODELS


def test_reader_tests_report_quartile_rates_and_both_hypotheses():
    rng = np.random.default_rng(0)
    rows = []
    for model in MODELS:
        for i in range(200):
            q = i % 4
            rows.append({"model": model, "quartile": q, "log_peak": q + rng.normal(0, 0.1),
                         "peakedness": rng.normal(), "density": rng.normal(), "token_purity": rng.normal(),
                         "network": float(rng.random() < (0.2 if q == 3 else 0.5))})
    out = reader_tests(rows, "network", 0)
    assert out["n"] == 1200 and out["by_quartile"][3] < out["by_quartile"][0]
    assert out["h1"]["holds"] and out["h2"]["holds"]


def test_with_coherence_moves_the_effect_onto_the_variable_that_carries_it():
    from recgen.analyze_v12 import with_coherence

    rng = np.random.default_rng(1)
    rows = []
    for model in MODELS:
        for _ in range(300):
            coherence = rng.normal()
            rows.append({"model": model, "log_peak": coherence + rng.normal(0, 0.8), "peakedness": rng.normal(),
                         "density": rng.normal(), "token_purity": rng.normal(), "coherence": coherence,
                         "haiku": float(rng.random() < 1 / (1 + np.exp(2 * coherence)))})
    out = with_coherence(rows, "haiku", np.random.default_rng(2))
    assert out["coherence"]["interval"][1] < 0
    assert abs(out["log_peak"]["coefficient"]) < abs(out["coherence"]["coefficient"]) / 3


def test_by_top50_mean_uses_the_given_quartile_thresholds():
    from recgen.analyze_v12 import by_top50_mean

    rows = [{"model": model, "log_top50_mean": float(v), "network": float(v < 3)} for model in MODELS for v in range(4)
            for _ in range(10)]
    thresholds = {model: np.array([0.5, 1.5, 2.5]) for model in MODELS}
    out = by_top50_mean(rows, "network", thresholds, np.random.default_rng(0))
    assert out["by_quartile"] == [1.0, 1.0, 1.0, 0.0] and out["h1"]["top_minus_rest"] == -1.0
