import numpy as np

import recgen.band_v20 as band_v20
from recgen.band_v17 import sample_band
from recgen.band_v20 import rank_band_examples, rank_windows
from recgen.rubric_v20 import BAND_SIZE, rank_rng


def test_rank_windows_take_the_ranks_inclusively_with_ties_broken_by_window():
    column = np.array([5.0, 9.0, 7.0, 7.0, 1.0, 3.0], dtype=np.float32)  # ranks: 1, 2, 3, 0, 5, 4
    windows, fires = rank_windows(column, np.array([], dtype=int), (2, 4))
    assert windows.tolist() == [2, 3, 0] and fires


def test_rank_windows_leave_out_the_top_windows_and_report_non_firing_bands():
    column = np.array([5.0, 9.0, 7.0, 0.0, -0.1, 3.0], dtype=np.float32)
    windows, fires = rank_windows(column, np.array([2]), (2, 5))
    assert windows.tolist() == [0, 5, 3] and not fires  # rank 5 is window 3 at 0.0


def test_rank_band_examples_sample_the_band_at_the_window_maximum(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_windows, n_neurons = 3000, 4
    window_max = rng.uniform(0, 2, size=(n_windows, n_neurons)).astype(np.float16)
    window_max[:, 3] = np.where(np.arange(n_windows) < 300, 1.0, 0.0)  # neuron 3 stops firing at rank 301
    pos = rng.integers(32, 512, size=(n_windows, n_neurons)).astype(np.int16)
    np.savez(tmp_path / "v5.npz", max=window_max, pos=pos)
    sampled = np.array([0, 2, 3])
    top_rows = np.stack([np.argsort(-window_max[:, n].astype(np.float32), kind="stable")[:50] for n in sampled])
    keys = np.stack([np.arange(n_windows), np.full(n_windows, 40)], axis=1)
    np.savez(tmp_path / "v6.npz", neurons=sampled, keys=keys, example_rows=top_rows)
    monkeypatch.setattr(band_v20, "sweep_path", lambda model, depth: tmp_path / "v5.npz")
    monkeypatch.setattr(band_v20, "v6_path", lambda model, depth: tmp_path / "v6.npz")

    out = rank_band_examples("70m", "band")
    assert out["included"].tolist() == [True, True, False] and out["counts"].tolist() == [300, 300, 300]
    assert np.all(out["keys"][2] == -1) and np.all(np.isnan(out["fractions"][2]))
    for s, n in enumerate(sampled[:2]):
        column = window_max[:, n].astype(np.float32)
        order = np.argsort(-column, kind="stable")
        windows = out["keys"][s][:, 0]
        assert len(set(windows.tolist())) == BAND_SIZE and set(windows.tolist()) <= set(order[200:500].tolist())
        assert np.array_equal(out["keys"][s][:, 1], pos[windows, n])
        assert np.allclose(out["fractions"][s], column[windows] / column.max())
        assert np.array_equal(windows, sample_band(order[200:500], rank_rng("70m", int(n), "band")))
    second = rank_band_examples("70m", "band2")
    assert second["included"].tolist() == [True, True, False]
    assert not set(second["keys"][0][:, 0].tolist()) & set(out["keys"][0][:, 0].tolist())
