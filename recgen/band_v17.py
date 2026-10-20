"""v17's band examples (recgen/rubric_v17.py): for each sampled neuron at the middle layer, BAND_SIZE
windows drawn uniformly from those whose window maximum is in BAND of its peak, outside its top 50,
one example per window at the window's strongest position. Deterministic, from v5's window maxima and
v6's recorded top-50 examples."""

import numpy as np

from recgen.rubric_v17 import BAND, BAND_SIZE, DEPTH, band_rng
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path


def band_windows(column: np.ndarray, top_windows: np.ndarray, band: tuple[float, float] = BAND) -> np.ndarray:
    """Windows whose maximum is at least band[0] and below band[1] times the peak of column [W], leaving
    out top_windows."""
    peak = column.max()
    in_band = (column >= band[0] * peak) & (column < band[1] * peak)
    in_band[top_windows] = False
    return np.flatnonzero(in_band)


def sample_band(windows: np.ndarray, rng, size: int = BAND_SIZE) -> np.ndarray | None:
    """size of the windows drawn uniformly without replacement, sorted, or None if there are fewer."""
    if len(windows) < size:
        return None
    return np.sort(rng.choice(windows, size=size, replace=False))


def band_examples(model: str) -> dict:
    """Every sampled neuron's band sample at DEPTH: neurons [S] (v6's order), keys [S, BAND_SIZE, 2]
    as (window, position) with -1 for excluded neurons, counts [S] of band windows, included [S]."""
    v6 = np.load(v6_path(model, DEPTH))
    neurons, top_windows = v6["neurons"], v6["keys"][v6["example_rows"]][:, :, 0]
    data = np.load(sweep_path(model, DEPTH))
    window_max, window_pos = data["max"], data["pos"]
    keys = np.full((len(neurons), BAND_SIZE, 2), -1, dtype=np.int32)
    counts = np.zeros(len(neurons), dtype=np.int32)
    for s, n in enumerate(neurons):
        windows = band_windows(window_max[:, n].astype(np.float32), top_windows[s])
        counts[s] = len(windows)
        chosen = sample_band(windows, band_rng(model, int(n)))
        if chosen is not None:
            keys[s] = np.stack([chosen, window_pos[chosen, n]], axis=1)
    return {"neurons": neurons, "keys": keys, "counts": counts, "included": counts >= BAND_SIZE}
