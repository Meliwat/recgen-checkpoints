import numpy as np
import pytest

import recgen.analyze_v6 as analyze_v6
from recgen.analyze_v6 import Layer, NullSpace, checks, decisions, mean_share, trend
from recgen.rubric_v5 import AGAINST, DEPTHS, FITS, MODELS, NO_TREND
from recgen.rubric_v6 import MERGE_PERCENTILE, embed, find_facets, least_related, pair_similarities, thresholds


@pytest.mark.parametrize("n_directions", [0, 1, 2])
def test_null_space_gives_the_rubrics_pair_similarities(n_directions):
    rng = np.random.default_rng(0)
    random, mean, scale = rng.normal(size=(60, 10)), rng.normal(size=10), rng.uniform(0.5, 2, size=10)
    directions = rng.normal(size=(n_directions, 10)) if n_directions else None
    fast = NullSpace((random - mean) * scale).similarities(directions)
    np.testing.assert_allclose(fast, pair_similarities(embed(random, mean, directions, scale)), atol=1e-12)


@pytest.fixture
def layer_file(tmp_path, monkeypatch):
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
    return np.load(path)


def test_layer_neuron_follows_the_rubric_definition(layer_file):
    result = Layer("70m", "middle").neuron(0)
    z, random = layer_file["z"].astype(np.float64), layer_file["z"][layer_file["random_rows"]].astype(np.float64)
    mean, direction = random.mean(axis=0), layer_file["weights"][0] * layer_file["gain"]
    units = embed(z[layer_file["example_rows"][0]], mean, direction)
    merge, unrelated = thresholds(pair_similarities(embed(random, mean, direction)))
    facets = find_facets(units, merge)
    assert result["n_facets"] == len(facets) == 2  # the two planted groups
    assert result["merge_level"] == pytest.approx(merge) and result["unrelated_level"] == pytest.approx(unrelated)
    assert result["least_related_similarity"] == pytest.approx(least_related(units, facets)[0])
    assert result["unrelated_mixed"] == (least_related(units, facets)[0] < unrelated)
    assert set(result["variants"]) == {"with LayerNorm gain", "uncentered", "unprojected"}


def test_mixture_pools_two_neurons(layer_file):
    assert isinstance(Layer("70m", "middle").mixture((0, 2)), bool)


def rows(shares):
    return [{"share": s, "interval": [s - 0.01, s + 0.01]} for s in shares]


def test_trend_judges_all_six_sizes_and_without_70m():
    t = trend(rows([0.30, 0.20, 0.20, 0.20, 0.20, 0.20]))
    assert t["verdict"] == NO_TREND and t["verdict_without_70m"] == NO_TREND
    t = trend(rows([0.30, 0.25, 0.20, 0.15, 0.10, 0.05]))
    assert t["verdict"] == FITS and t["verdict_without_70m"] == FITS


def fake_stats(unrelated, power):
    stats = []
    for i, model in enumerate(MODELS):
        for depth in DEPTHS:
            check = {"v6": 0.1, "v5": 0.2}
            stats.append({"model": model, "depth": depth, "unrelated_mixed": rows([unrelated[i]])[0],
                          "multi_facet": rows([0.5])[0],
                          "checks": {"mixture_power": {"v6": rows([power[i]])[0], "v5": rows([0.5])[0]},
                                     "no_facet_share": check, "median_top_pair_percentile": check,
                                     "median_token_purity": check}})
    return stats


def test_decisions_say_whether_v5s_verdicts_hold():
    falling = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    v5_layers = [{"model": m, "depth": d, "unrelated_mixed_share": s, "unrelated_mixed_interval": [s - 0.01, s + 0.01],
                  "multi_facet_share": 0.5, "multi_facet_interval": [0.49, 0.51]}
                 for m, s in zip(MODELS, falling[::-1]) for d in DEPTHS]
    v5_decisions = {f"{d}: {m}": {"verdict": AGAINST if m == "unrelated_mixed" else NO_TREND}
                    for d in DEPTHS for m in ("unrelated_mixed", "multi_facet")}
    out = decisions(fake_stats(falling, falling), v5_layers, v5_decisions)
    primary = out["middle: unrelated_mixed"]
    assert primary["primary"] and primary["v6"]["verdict"] == FITS and primary["v5"]["verdict"] == AGAINST
    assert not primary["v5_verdict_holds"] and out["middle: multi_facet"]["v5_verdict_holds"]


def test_checks_read_falling_power_as_lost_power():
    out = checks(fake_stats([0.2] * 6, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
    assert out["middle: v6"]["mixture_power"]["verdict"] == "falls with size"
    assert out["middle: v5"]["mixture_power"]["verdict"] == NO_TREND


def test_mean_share_is_the_part_of_the_norm_shared_by_every_vector():
    vectors = np.array([[1.0, 1.0], [1.0, -1.0]])  # common part (1, 0), the rest cancels
    assert mean_share(vectors) == pytest.approx(0.5)
    assert mean_share(np.array([[1.0, 0.0], [-1.0, 0.0]])) == 0.0
