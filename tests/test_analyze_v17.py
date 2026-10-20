import numpy as np
import pytest

import recgen.analyze_v6 as analyze_v6
from recgen.analyze_v6 import Layer
from recgen.analyze_v17 import agreement, neuron_readings, reader_tests
from recgen.rubric_v5 import MODELS
from recgen.rubric_v17 import MORE_MIXED, PERSISTS, TOP50_SPECIFIC


@pytest.fixture
def layer(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    d, n_random = 12, 150
    centers = rng.normal(size=(2, d)) * 3
    examples = np.vstack([centers[0] + rng.normal(size=(30, d)), centers[1] + rng.normal(size=(30, d))])
    z = np.vstack([rng.normal(size=(n_random, d)), examples]).astype(np.float16)
    example_rows = np.stack([n_random + rng.permutation(60)[:50] for _ in range(3)]).astype(np.int32)
    path = tmp_path / "layer.npz"
    np.savez(path, keys=np.stack([np.arange(len(z)), np.full(len(z), 40)], axis=1), z=z, neurons=np.array([4, 9, 11]),
             example_rows=example_rows, random_rows=np.arange(n_random, dtype=np.int32),
             weights=rng.normal(size=(3, d)).astype(np.float32), gain=rng.uniform(0.5, 1.5, size=d).astype(np.float32),
             constant=np.zeros(3, np.float32), layer=3, n_layers=6, activation_error=0.001)
    monkeypatch.setattr(analyze_v6, "v6_path", lambda model, depth: path)
    return Layer("70m", "middle")


def test_network_readings_match_v6s_detector_on_any_sample(layer):
    random_rows = np.arange(50)
    for s in range(3):
        v6 = layer.neuron(s)
        # the "band" here is the top 50 again, read from a separate residual array, as v17's band is
        r = neuron_readings(layer, layer.z.copy(), layer.rows[s], random_rows, s)
        for sample in ("top", "band"):
            assert r[sample]["flag"] == v6["unrelated_mixed"] and r[sample]["n_facets"] == v6["n_facets"]
            assert r[sample]["coherence"] == v6["top_pair_percentile"]
        assert r["reference"]["n_facets"] == 0  # random positions form no facet
    assert neuron_readings(layer, layer.z, np.full(50, -1), random_rows, 0)["band"] is None


def synthetic_rows(rng, band_effect: float) -> list[dict]:
    """Top-50 flags with a magnitude effect (strongest quartile 0.2, the rest 0.5); band flags higher by 0.2,
    with the strongest quartile at 0.7 - band_effect; coherence 20 points lower in the band."""
    rows = []
    for model in MODELS:
        for i in range(400):
            q = i % 4
            top_rate, band_rate = (0.2, 0.7 - band_effect) if q == 3 else (0.5, 0.7)
            coherence = 80 + 5 * q + rng.normal(0, 3)
            rows.append({"model": model, "quartile": q, "log_peak": q + rng.normal(0, 0.1),
                         "top_flag": float(rng.random() < top_rate), "band_flag": float(rng.random() < band_rate),
                         "top_coherence": coherence, "band_coherence": coherence - 20 + rng.normal(0, 3)})
    return rows


def test_reader_tests_read_t1_and_the_verdict():
    rng = np.random.default_rng(0)
    persists = reader_tests(synthetic_rows(rng, band_effect=0.3), "flag", 0)
    assert persists["t1"]["reading"] == MORE_MIXED and persists["t2"]["verdict"] == PERSISTS
    assert set(persists["by_quartile"]) == {"top", "band"} and persists["by_quartile"]["top"][3] < 0.3
    vanishes = reader_tests(synthetic_rows(rng, band_effect=0.0), "flag", 0)
    assert vanishes["t2"]["verdict"] == TOP50_SPECIFIC
    coherence = reader_tests(synthetic_rows(rng, band_effect=0.0), "coherence", 0)
    assert coherence["t1"]["reading"] == MORE_MIXED and coherence["t1"]["band_minus_top"] < -15
    assert coherence["t2"]["verdict"] == PERSISTS  # stronger neurons stay more coherent in the band


def test_agreement_counts_labels_and_flags():
    out = agreement(["A", "C", "B", "C"], ["A", "C", "A", "A"])
    assert out["same_label"] == 0.5 and out["same_flag"] == 0.75 and out["flag_rates"] == [0.5, 0.25]
