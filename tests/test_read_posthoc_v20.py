import numpy as np

import recgen.read_posthoc_v20 as posthoc
from recgen.read_posthoc_v20 import POOL_PART, POOLS_PER_QUARTILE, RANDOM_UNITS, pool_picks, pool_units, random_units
from recgen.rubric_v5 import MODELS
from recgen.rubric_v20 import BAND_SIZE


def test_pool_picks_take_part_of_each_neuron_without_a_shared_position():
    keys_a = np.stack([np.arange(BAND_SIZE), np.full(BAND_SIZE, 40)], axis=1)
    keys_b = keys_a.copy()
    keys_b[::2, 0] += 1000  # half of b's positions differ from a's, half are a's
    from_a, from_b = pool_picks(keys_a, keys_b, np.random.default_rng(0))
    assert len(set(from_a.tolist())) == len(set(from_b.tolist())) == POOL_PART
    assert not {tuple(k) for k in keys_a[from_a].tolist()} & {tuple(k) for k in keys_b[from_b].tolist()}
    again = pool_picks(keys_a, keys_b, np.random.default_rng(0))
    assert np.array_equal(again[0], from_a) and np.array_equal(again[1], from_b)


def test_pool_units_pair_distinct_neurons_within_a_quartile(monkeypatch):
    fresh = [{"neurons": (s,), "quartile": s % 4} for s in range(100)]
    keys = np.stack([np.stack([np.arange(BAND_SIZE) + 100 * s, np.full(BAND_SIZE, s)], axis=1) for s in range(100)])
    monkeypatch.setattr(posthoc, "fresh_units", lambda model: fresh)
    monkeypatch.setattr(posthoc, "rank_band_examples",
                        lambda model, band: {"keys": keys, "included": np.ones(100, dtype=bool)})
    units = pool_units("70m")
    assert len(units) == 4 * POOLS_PER_QUARTILE
    neurons = [s for u in units for s in u["neurons"]]
    assert len(set(neurons)) == len(neurons) and all(s % 4 == u["quartile"] for u in units for s in u["neurons"])
    u = units[0]
    (a, b), (from_a, from_b) = u["neurons"], u["picks"]
    assert np.array_equal(u["keys"], np.concatenate([keys[a][from_a], keys[b][from_b]])) and len(u["keys"]) == BAND_SIZE
    assert [x["id"] for x in pool_units("70m")] == [x["id"] for x in units] != [x["id"] for x in pool_units("160m")]


def test_random_units_spread_over_the_sizes_one_position_per_window(tmp_path, monkeypatch):
    keys = np.stack([np.arange(1024) * 4, np.full(1024, 100)], axis=1)
    np.savez(tmp_path / "v6.npz", keys=np.concatenate([keys[::-1], keys]), random_rows=np.arange(1024, 2048))
    monkeypatch.setattr(posthoc, "v6_path", lambda model, depth: tmp_path / "v6.npz")
    units = {model: random_units(model) for model in MODELS}
    assert sum(len(u) for u in units.values()) == RANDOM_UNITS and {len(u) for u in units.values()} == {16, 17}
    for u in units["70m"]:
        assert len(set(u["keys"][:, 0].tolist())) == BAND_SIZE and set(u["keys"][:, 0].tolist()) <= set(keys[:, 0].tolist())
    assert not np.array_equal(units["70m"][0]["keys"], units["70m"][1]["keys"])
