import numpy as np

import recgen.analyze_posthoc_v20 as posthoc
from recgen.analyze_posthoc_v20 import excess_by_quartile, kappa_interval, network_pool_flags, random_floor, same_round
from recgen.analyze_v6 import PRIMARY
from recgen.rubric_v5 import MODELS


def haiku_rows(rng) -> list[dict]:
    """v12's top-50 flags with a magnitude effect (strongest quartile 0.1, the rest 0.5) and rank-band flags with a
    smaller one (0.3 against 0.5)."""
    rows = []
    for model in MODELS:
        for i in range(200):
            q = i % 4
            rows.append({"model": model, "neuron": i, "quartile": q, "log_peak": q + rng.normal(0, 0.1),
                         "top_flag": float(rng.random() < (0.1 if q == 3 else 0.5)),
                         "band_flag": float(rng.random() < (0.3 if q == 3 else 0.5)),
                         "band2_flag": None if i == 0 else float(rng.random() < 0.5)})
    return rows


def test_same_round_replaces_v12s_top_50_flags_with_this_rounds(monkeypatch):
    rows = haiku_rows(np.random.default_rng(0))
    monkeypatch.setattr(posthoc, "fresh_units", lambda model: [{"id": f"{model}:fresh::{i}", "neuron_id": i} for i in range(200)])
    # this round's top-50 reading of each neuron is its rank-band reading, except one neuron left without a label
    labels = {f"{r['model']}:fresh::{r['neuron']}": "C" if r["band_flag"] else "A" for r in rows}
    labels["70m:fresh::1"] = None
    answers = {"top_same_round": {uid: {"label": label} for uid, label in labels.items()}}
    reread = {f"70m:fresh::{i}": {"label": labels[f"70m:fresh::{i}"]} for i in (2, 3, 4)}
    out = same_round(rows, answers, {"top_reread": reread})
    assert out["n"] == len(rows) - 1 and out["band"]["n"] == len(rows) - 1 and out["band2"]["n"] == len(rows) - 1 - len(MODELS)
    assert out["band"]["t1"]["band_minus_top"] == 0 and out["band"]["t2_band_minus_top"]["difference"] == 0
    assert out["by_quartile"]["top_same_round"] == out["by_quartile"]["band"]
    drift = out["drift_same_round_minus_v12"]
    assert drift["by_quartile"][3]["mean"] > 0.1 and drift["interval"][0] <= drift["mean_over_models"] <= drift["interval"][1]
    assert out["top_read_twice_this_round"]["n"] == 3 and out["top_read_twice_this_round"]["same_label"] == 1


def test_excess_by_quartile_compares_pools_with_single_neurons_of_their_quartile():
    pools = [{"quartile": q, "flag": float(i < (10 if q == 3 else 5)), "constituents": 0.5} for q in range(4) for i in range(10)]
    single = [{"quartile": q, "band_flag": float(i < 50)} for q in range(4) for i in range(100)]
    out = excess_by_quartile(pools, single, np.random.default_rng(0))
    assert out["pools"] == [10] * 4 and out["single_rate"] == [0.5] * 4 and out["pool_rate"] == [0.5, 0.5, 0.5, 1.0]
    assert [e["difference"] for e in out["excess"]] == [0.0, 0.0, 0.0, 0.5] and out["constituents_rate"] == [0.5] * 4
    low, high = out["excess"][0]["interval"]
    assert low < 0 < high and out["excess"][3]["interval"][0] > 0
    assert out["strongest_minus_weakest_excess"]["difference"] == 0.5


def test_kappa_interval_is_one_for_identical_readings():
    labels = ["A", "C", "B", "C"] * 5
    assert kappa_interval(labels, labels, np.random.default_rng(0)) == [1.0, 1.0]


def test_random_floor_reads_the_random_units_and_the_network_reference(monkeypatch):
    monkeypatch.setattr(posthoc, "random_units", lambda model: [{"id": f"{model}:random::{i}"} for i in range(2)])
    answers = {"random": {f"{m}:random::{i}": {"label": "C" if i else ("A" if m == "70m" else None)}
                          for m in MODELS for i in range(2)}}
    network = [{"reference_flag": float(i == 0), "reference_coherence": 50.0} for i in range(40)]
    out = random_floor(answers, network)
    assert out["haiku_flag"]["n"] == 7 and abs(out["haiku_flag"]["mean"] - 6 / 7) < 1e-12
    assert out["haiku_label_shares"]["A"] == 1 / 7
    assert out["network_reference_flag"]["mean"] == 1 / 40 and out["network_reference_coherence"]["mean"] == 50


def test_network_pool_flags_project_out_both_neurons_directions(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    d, n_rows = 8, 120
    directions = rng.normal(size=(3, d))

    class FakeNulls:
        def similarities(self, pair):
            assert np.array_equal(pair, directions[[0, 2]])
            return np.zeros(10)

    class FakeLayer:
        def __init__(self, model, depth):
            self.embeddings = {PRIMARY: (np.zeros(d), None, directions)}
            self.nulls = {PRIMARY: FakeNulls()}

    keys = np.stack([np.arange(n_rows), np.full(n_rows, 40)], axis=1)
    band_rows = np.stack([np.arange(50), np.arange(50) + 50, np.arange(50) + 70])
    np.savez(tmp_path / "v20.npz", z=rng.normal(size=(n_rows, d)), band_rows=band_rows, keys=keys)
    seen = []

    def fake_read(units, null):
        seen.append(units)
        return {"flag": 1.0}

    monkeypatch.setattr(posthoc, "Layer", FakeLayer)
    monkeypatch.setattr(posthoc, "v20_path", lambda model: tmp_path / "v20.npz")
    monkeypatch.setattr(posthoc, "read_examples", fake_read)
    picks = (np.arange(25), np.arange(25, 50))
    unit = {"neurons": (0, 2), "picks": picks, "keys": keys[np.concatenate([band_rows[0][:25], band_rows[2][25:]])]}
    assert network_pool_flags("70m", [unit]) == [1.0]
    assert seen[0].shape == (50, d) and np.allclose(seen[0] @ directions[[0, 2]].T, 0)
