import numpy as np
import pytest

from recgen.analyze_v6 import PRIMARY, Layer, detect
from recgen.position_v19 import (
    POSITION_FEATURES, VARIANTS, WithoutSubspace, null_position_correlation, position_features, position_fit,
    random_subspace, recorded_null, summarize, without,
)
from recgen.rubric_v6 import MERGE_PERCENTILE, embed, pair_similarities, thresholds
from recgen.rubric_v19 import MODELS, excess_over_models, h1


def test_position_features_are_smooth_functions_of_the_window_fraction():
    x = position_features(np.array([32, 256, 511]))
    assert x.shape == (3, len(POSITION_FEATURES)) == (3, 10)
    np.testing.assert_allclose(x[:, 0], 1)
    np.testing.assert_allclose(x[1, 1:4], [0.5, 0.25, np.log(0.5)])
    np.testing.assert_allclose(x[1, 4:6], [np.sin(np.pi), np.cos(np.pi)], atol=1e-12)


def test_position_fit_finds_the_directions_position_moves_and_its_r2():
    rng = np.random.default_rng(0)
    d, n = 16, 3000
    positions = rng.integers(32, 512, n)
    p = positions / 512
    u, v = np.linalg.qr(rng.normal(size=(d, 2)))[0].T
    z = rng.normal(size=(n, d)) + 4 * np.outer(np.sin(2 * np.pi * p), u) + 3 * np.outer(p, v) + 5
    vt, captured, r2 = position_fit(z.astype(np.float16), positions)
    top2 = vt[:2]
    assert np.allclose(np.linalg.norm(top2 @ np.stack([u, v]).T, axis=0), 1, atol=0.02)  # u and v lie in the top 2
    assert captured[1] > 0.99 and captured[-1] == pytest.approx(1)
    centered = z.astype(np.float16).astype(np.float64) - z.astype(np.float16).astype(np.float64).mean(axis=0)
    coef = np.linalg.lstsq(position_features(positions), centered, rcond=None)[0]
    residual = centered - position_features(positions) @ coef
    assert r2 == pytest.approx(1 - np.sum(residual ** 2) / np.sum(centered ** 2))


def test_random_subspace_is_orthonormal_and_without_projects_it_out():
    basis = random_subspace(12, 3, np.random.default_rng(0))
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    x = np.random.default_rng(1).normal(size=(5, 12))
    np.testing.assert_allclose(without(x, basis) @ basis.T, 0, atol=1e-12)


def test_null_position_correlation_is_negative_when_position_drives_similarity():
    rng = np.random.default_rng(2)
    positions = rng.integers(32, 512, 300)
    p = positions / 512
    plain = rng.normal(size=(300, 20))
    assert abs(null_position_correlation(plain - plain.mean(axis=0), positions)) < 0.05
    drift = plain + 6 * np.stack([np.sin(np.pi * p), np.cos(np.pi * p)] + [np.zeros_like(p)] * 18, axis=1)
    assert null_position_correlation(drift - drift.mean(axis=0), positions) < -0.3


@pytest.fixture
def layer_path(tmp_path):
    rng = np.random.default_rng(1)
    d, n_random = 12, 150
    centers = rng.normal(size=(2, d)) * 3
    examples = np.vstack([centers[0] + rng.normal(size=(30, d)), centers[1] + rng.normal(size=(30, d))])
    z = np.vstack([rng.normal(size=(n_random, d)), examples]).astype(np.float16)
    path = tmp_path / "layer.npz"
    np.savez(path, keys=np.stack([np.arange(len(z)), rng.integers(32, 512, len(z))], axis=1), z=z,
             neurons=np.array([4, 9, 11]), example_rows=np.stack([n_random + rng.permutation(60)[:50] for _ in range(3)]),
             random_rows=np.arange(n_random), weights=rng.normal(size=(3, d)).astype(np.float32),
             gain=rng.uniform(0.5, 1.5, size=d).astype(np.float32), constant=np.zeros(3, np.float32), layer=3,
             n_layers=6, activation_error=0.001)
    return path


def test_without_an_empty_subspace_the_reader_is_the_layers_own(layer_path):
    layer = Layer("gpt2", "middle", layer_path)
    reader = WithoutSubspace(layer, np.zeros((0, layer.z.shape[1])))
    for s in range(len(layer.neurons)):
        units, _, lv = layer.embedded(PRIMARY, layer.rows[s], [s])
        assert reader.flag(layer.rows[s], [s]) == detect(units, lv["merge"][MERGE_PERCENTILE], lv["unrelated"])["unrelated_mixed"]


def test_the_subspace_is_projected_out_of_examples_and_null_with_the_neuron_directions(layer_path):
    layer = Layer("gpt2", "middle", layer_path)
    data = np.load(layer_path)
    random = data["z"][data["random_rows"]].astype(np.float64)
    mean = random.mean(axis=0)
    basis = random_subspace(layer.z.shape[1], 2, np.random.default_rng(3))
    removed = np.vstack([data["weights"][[0, 2]].astype(np.float64) * data["gain"].astype(np.float64), basis])
    reader = WithoutSubspace(layer, basis)
    expected_null = pair_similarities(embed(random, mean, removed))
    np.testing.assert_allclose(reader.null.similarities(reader.directions[[0, 2]]), expected_null, atol=1e-10)
    rows = np.concatenate([layer.rows[0][:25], layer.rows[2][:25]])
    merge, unrelated = thresholds(expected_null)
    expected = detect(embed(data["z"][rows], mean, removed), merge, unrelated)["unrelated_mixed"]
    assert reader.flag(rows, [0, 2]) == expected


def test_recorded_null_is_centered_with_its_positions(layer_path):
    null, positions = recorded_null(layer_path)
    data = np.load(layer_path)
    np.testing.assert_allclose(null.mean(axis=0), 0, atol=1e-12)
    assert np.array_equal(positions, data["keys"][:150, 1])


def test_summarize_compares_new_flags_with_the_originals_on_the_original_draws():
    rng = np.random.default_rng(4)
    by_model, neurons, units = {}, {}, {}
    for model in MODELS:
        old = (rng.random(400) < 0.4).astype(float)
        neurons[model] = [{"network": f, "quartile": i % 4, "log_peak": float(i % 4)} for i, f in enumerate(old)]
        units[model] = ([{"group": "real", "neurons": (s,)} for s in range(0, 200, 2)]
                        + [{"group": "pooled 50/50", "network": True} for _ in range(50)])
        new = old.copy()
        new[:40] = 1 - new[:40]
        by_model[model] = {"variants": {name: {"neuron_flags": new, "pooled_flags": np.r_[np.ones(30), np.zeros(20)]}
                                        for name in VARIANTS}}
    out = summarize(by_model, neurons, units, lambda: np.random.default_rng(7), lambda: np.random.default_rng(8))
    for name in VARIANTS:
        assert out[name]["neuron_flags_changed"] == pytest.approx(0.1)
        first = out[name]["by_model"][MODELS[0]]
        assert first["neuron_flags_unchanged"] == pytest.approx(0.9) and first["pooled_50_50_flagged"] == [1.0, 0.6]
        new = by_model[MODELS[0]]["variants"][name]["neuron_flags"]
        assert first["real_flagged"][1] == pytest.approx(new[0:200:2].mean())
        pairs = [(by_model[m]["variants"][name]["pooled_flags"], by_model[m]["variants"][name]["neuron_flags"][0:200:2])
                 for m in MODELS]
        assert out[name]["network_excess"]["interval"] == excess_over_models(pairs, np.random.default_rng(8))["interval"]
        groups = [(np.array([n["log_peak"] for n in neurons[m]]), np.array([n["quartile"] for n in neurons[m]]),
                   by_model[m]["variants"][name]["neuron_flags"]) for m in MODELS]
        assert out[name]["h1"] == h1(groups, np.random.default_rng(7))
