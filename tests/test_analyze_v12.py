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
