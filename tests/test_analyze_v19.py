import numpy as np
import pytest

from recgen.analyze_v19 import GPT2MiniLM, excess_by_reader, reader_tests
from recgen.rubric_v19 import MODELS


def unit(rng, center: np.ndarray, n: int) -> np.ndarray:
    v = center + rng.normal(0, 0.05, (n, len(center)))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_gpt2_minilm_applies_v5s_rules_to_cached_vectors():
    rng = np.random.default_rng(0)
    a, b = np.eye(8)[0], np.eye(8)[1]
    vectors = np.concatenate([unit(rng, a, 25), unit(rng, b, 25), unit(rng, a, 50)])
    keys = np.stack([np.arange(100), np.full(100, 40)], axis=1)
    minilm = GPT2MiniLM(keys, vectors.astype(np.float16), rng.normal(size=(64, 8)))
    assert minilm.detect(keys[:50])["unrelated_mixed"]  # two orthogonal groups of 25
    assert not minilm.detect(keys[50:])["unrelated_mixed"]  # one group
    with pytest.raises(AssertionError):
        minilm.detect(np.array([[5, 41]]))  # an example that was never embedded
    assert set(minilm.calibration()) == {"0.3", "0.25", "0.2", "0.1"}


def test_excess_by_reader_averages_models_and_skips_missing_labels():
    units = {}
    for model in MODELS:
        rows = [{"group": "real", "minilm": False, "network": i < 40, "llm": i < 30} for i in range(100)]
        rows += [{"group": "pooled 50/50", "minilm": False, "network": i < 35, "llm": i < 30 if i < 49 else None}
                 for i in range(50)]
        units[model] = rows
    out = excess_by_reader(units, 0)
    assert out["minilm"]["excess"] == 0 and abs(out["network"]["excess"] - 0.3) < 1e-12
    assert abs(out["llm"]["excess"] - (30 / 49 - 0.3)) < 1e-12 and out["network"]["interval"][0] > 0


def test_reader_tests_run_v12s_hypotheses_over_the_three_models():
    rng = np.random.default_rng(1)
    rows = []
    for model in MODELS:
        for i in range(400):
            q = i % 4
            rows.append({"model": model, "quartile": q, "log_peak": q + rng.normal(0, 0.1), "peakedness": rng.normal(),
                         "density": rng.normal(), "token_purity": rng.normal(),
                         "llm": None if i == 0 else float(rng.random() < (0.15 if q == 3 else 0.5))})
    out = reader_tests(rows, "llm", 0)
    assert out["n"] == 1197 and set(out["by_quartile_by_model"]) == set(MODELS)
    assert out["h1"]["holds"] and out["h1"]["models_negative"] == 3 and out["h2"]["holds"]
