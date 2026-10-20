"""v20's rank-band examples (recgen/rubric_v20.py): for each sampled neuron at the middle layer and each band,
BAND_SIZE windows drawn uniformly from those at the band's ranks of the neuron's own window maxima, outside its
top 50, one example per window at the window's strongest position. Deterministic, from v5's window maxima and
v6's recorded top-50 examples. Each example's fraction of the neuron's peak is kept for the comparison with v17."""

import numpy as np

from recgen.band_v17 import sample_band
from recgen.rubric_v20 import BAND_SIZE, DEPTH, RANKS, rank_rng
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path


def ranked(column: np.ndarray) -> np.ndarray:
    """Windows of column [W] from largest to smallest maximum, ties broken by window index."""
    return np.argsort(-column, kind="stable")


def rank_windows(column: np.ndarray, top_windows: np.ndarray, ranks: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """The windows ranked ranks[0] to ranks[1] (inclusive, 1 is the largest) of column [W], leaving out
    top_windows, and whether the neuron fires in all of them (its maximum at the last rank is above 0)."""
    order = ranked(column)
    first, last = ranks
    windows = order[first - 1:last]
    return windows[~np.isin(windows, top_windows)], bool(column[order[last - 1]] > 0)


def rank_band_examples(model: str, band: str) -> dict:
    """Every sampled neuron's sample from one band at DEPTH: neurons [S] (v6's order), keys [S, BAND_SIZE, 2]
    as (window, position) and fractions [S, BAND_SIZE] of the neuron's peak, with -1 and NaN for excluded
    neurons, counts [S] of band windows, included [S]."""
    v6 = np.load(v6_path(model, DEPTH))
    neurons, top_windows = v6["neurons"], v6["keys"][v6["example_rows"]][:, :, 0]
    data = np.load(sweep_path(model, DEPTH))
    window_max, window_pos = data["max"], data["pos"]
    keys = np.full((len(neurons), BAND_SIZE, 2), -1, dtype=np.int32)
    fractions = np.full((len(neurons), BAND_SIZE), np.nan, dtype=np.float32)
    counts = np.zeros(len(neurons), dtype=np.int32)
    included = np.zeros(len(neurons), dtype=bool)
    for s, n in enumerate(neurons):
        column = window_max[:, n].astype(np.float32)
        windows, fires = rank_windows(column, top_windows[s], RANKS[band])
        counts[s] = len(windows)
        chosen = sample_band(windows, rank_rng(model, int(n), band)) if fires else None
        if chosen is not None:
            keys[s] = np.stack([chosen, window_pos[chosen, n]], axis=1)
            fractions[s] = column[chosen] / column.max()
            included[s] = True
    return {"neurons": neurons, "keys": keys, "fractions": fractions, "counts": counts, "included": included}
