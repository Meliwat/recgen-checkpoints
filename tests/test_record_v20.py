import numpy as np
import pytest

import recgen.analyze_v6 as analyze_v6
from recgen.analyze_v6 import Layer
from recgen.analyze_v17 import neuron_readings
from recgen.record_v20 import band_readings


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


def test_band_readings_equal_v17s_band_readings_and_skip_excluded_neurons(layer):
    rng = np.random.default_rng(2)
    z = layer.z.copy()
    rows = np.stack([rng.choice(len(z), 50, replace=False), layer.rows[1], np.full(50, -1)])
    out = band_readings(layer, z, rows)
    for s in range(2):
        assert out[s] == neuron_readings(layer, z, rows[s], np.arange(50), s)["band"]
    assert out[2] is None
