import numpy as np

import recgen.band_v17 as band_v17
from recgen.band_v17 import band_examples, band_windows, sample_band
from recgen.rubric_v17 import BAND_SIZE, band_rng


def test_band_includes_its_low_edge_and_excludes_its_high_edge():
    column = np.array([10.0, 6.0, 5.999, 4.0, 3.999, 5.0], dtype=np.float32)
    assert band_windows(column, np.array([], dtype=int)).tolist() == [2, 3, 5]


def test_band_leaves_out_the_top_windows():
    column = np.array([10.0, 5.0, 5.0, 5.0], dtype=np.float32)
    assert band_windows(column, np.array([0, 2])).tolist() == [1, 3]


def test_sample_band_is_uniform_without_replacement_and_needs_enough_windows():
    windows = np.arange(100, 300)
    a, b = sample_band(windows, band_rng("70m", 3)), sample_band(windows, band_rng("70m", 3))
    assert np.array_equal(a, b) and len(set(a.tolist())) == BAND_SIZE and np.all(np.diff(a) > 0)
    assert set(a.tolist()) <= set(windows.tolist())
    assert sample_band(windows[:BAND_SIZE - 1], band_rng("70m", 3)) is None
    counts = np.zeros(200)
    for seed in range(400):
        counts[sample_band(windows, np.random.default_rng(seed)) - 100] += 1
    assert counts.min() > 50 and counts.max() < 150  # each window drawn about 400 * 50 / 200 = 100 times


def test_band_examples_take_the_window_maximum_position_and_exclude_sparse_neurons(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_windows, n_neurons = 400, 4
    window_max = rng.uniform(0, 1, size=(n_windows, n_neurons)).astype(np.float16)
    window_max[0] = 1.0  # every neuron peaks at 1 in window 0
    window_max[:, 3] = np.where(np.arange(n_windows) < 60, 1.0, 0.0)  # neuron 3 has no band windows
    pos = rng.integers(32, 512, size=(n_windows, n_neurons)).astype(np.int16)
    np.savez(tmp_path / "v5.npz", max=window_max, pos=pos)
    sampled = np.array([0, 1, 3])
    top_rows = np.stack([np.argsort(-window_max[:, n].astype(np.float32))[:50] for n in sampled])
    keys = np.stack([np.arange(n_windows), np.full(n_windows, 40)], axis=1)
    np.savez(tmp_path / "v6.npz", neurons=sampled, keys=keys, example_rows=top_rows)
    monkeypatch.setattr(band_v17, "sweep_path", lambda model, depth: tmp_path / "v5.npz")
    monkeypatch.setattr(band_v17, "v6_path", lambda model, depth: tmp_path / "v6.npz")

    out = band_examples("70m")
    assert out["included"].tolist() == [True, True, False] and out["counts"][2] == 0
    assert np.all(out["keys"][2] == -1)
    for s, n in enumerate(sampled[:2]):
        windows = out["keys"][s][:, 0]
        assert np.array_equal(out["keys"][s][:, 1], pos[windows, n])
        fraction = window_max[windows, n].astype(np.float32) / window_max[:, n].astype(np.float32).max()
        assert np.all((fraction >= 0.4) & (fraction < 0.6))
        assert not set(windows.tolist()) & set(top_rows[s].tolist())
        assert np.array_equal(windows, sample_band(band_windows(window_max[:, n].astype(np.float32), top_rows[s]),
                                                   band_rng("70m", int(n))))


def test_band_rows_index_the_unique_positions_and_mark_excluded_neurons():
    from recgen.record_v17 import band_rows

    keys = np.array([[[3, 40], [1, 100]], [[-1, -1], [-1, -1]], [[1, 100], [2, 33]]], dtype=np.int32)
    unique = np.unique(keys[[0, 2]].reshape(-1, 2), axis=0)  # (1, 100), (2, 33), (3, 40)
    rows = band_rows(keys, unique)
    assert rows.tolist() == [[2, 0], [-1, -1], [0, 1]]
