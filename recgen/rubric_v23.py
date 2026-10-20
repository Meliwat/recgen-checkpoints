"""Pre-registered v23 rules, written and committed before any v23 recording.

v12 found (pre-registered, held out) that neurons with a larger peak activation are flagged "unrelated
mixed" much less often by the network-native reader (v6) and by Haiku, and v17 and v20 found the effect
persists below the top 50. Every one of those tests judges how alike a neuron's examples are, so none can
tell "weakly firing neurons do several unrelated jobs" from "weakly firing neurons carry one broad, diffuse
feature". v2's driver test measures something else: whether two facets of a neuron are driven by the same
upstream MLP neurons. v23 asks, at Pythia-410m layer 12, final checkpoint: are the facets of weakly firing
neurons driven by different upstream circuits more often than the facets of strongly firing neurons?

Neurons. Peak is v12's: a neuron's largest window maximum (v5's sweep), and peak quartiles are v12's,
taken over all of v6's 2,048 sampled neurons of the layer. Eligible neurons are those whose 50 strongest
examples form at least MIN_FACETS facets under v6's detector at its pre-registered settings (v6's
n_facets, which the candidate builder checks it reproduces). PER_QUARTILE eligible neurons are drawn from
each peak quartile with seed [SEED, 23, quartile] (fewer if a quartile has fewer eligible; reported).
Before writing these rules, the eligible counts were checked (no recording): 467, 471, 461 and 410 by
quartile, weakest to strongest. Facets are v6's facets of the top 50, as in v13.

Recording. v4/v13's attribution (recgen.formation.capture: direct terms of each neuron's pre-activation
from every upstream MLP neuron and attention head, with the LayerNorm scale frozen) at the final checkpoint
(step 143000) only, on every facet example and on background text, to data/v23/. Two departures from v13,
for size and memory: examples are captured in chunks within one checkpoint load, and head terms are stored
in float16 like the MLP activations (the reconstruction error is checked in float32 before storing).
Background is v13's background positions, unchanged (v4's rule applied to v13's candidates, same seed):
v4's rule needs 1,024 windows that hold no facet example, and a pilot count (no recording) found the
union of 200 eligible neurons' top-50 windows covers 3,612 of the 4,096 windows, leaving too few. Sharing
v13's background also makes the v13 consistency check exact. A background position that happens to be a
v23 facet example stays in the background, as random text.

Per neuron, for MLP drivers (top 100, primary) and head drivers (top 5 of 192, beside them):
- every facet pair's driver overlap and v2's adjacency rule against this neuron's null: its facets against
  the other selected neurons' facets, read through its weights, 95th percentile (analyze_v13's
  final_adjacency). A pair is driver-unrelated if it is not adjacent (overlap at or below the threshold).
- primary measure: whether the least-overlapping pair is driver-unrelated (ties broken by pair order).
  Because one threshold serves all of a neuron's pairs, this is also "some pair is driver-unrelated".
- secondary: the share of its pairs that are driver-unrelated; the margin, the least-overlapping pair's
  overlap minus the threshold (continuous; higher means more shared); and whether the pair v6's reader
  calls least related is driver-unrelated.

Pre-registered tests (MLP drivers). Bootstrap intervals resample neurons within each peak quartile (the
sample is stratified by quartile), BOOTSTRAP draws, 2.5th to 97.5th percentiles.
- P1: the top peak quartile's rate of driver-unrelated least pairs minus the other three's.
- P2: Spearman correlation of peak with the margin.
- P3: P1 minus the same top-minus-rest difference in v6's unrelated-mixed flag on the same neurons
  (paired), which asks whether drivers follow the reader's gradient at its full size.
Reading (reading() below), checked in this order:
- "more jobs": P1's interval entirely below 0 and P2's entirely above 0. Weak neurons' facets are driven
  by different upstream circuits more often, which supports "weakly firing neurons do more unrelated jobs".
- "weak neurons share more": P1's interval entirely above 0 and P2's entirely below 0. Strong neurons'
  facets are the more separate ones, which supports "one broad feature" in weak neurons.
- "no driver gradient": P1's interval not entirely below 0, P2's not entirely above 0, and P3's entirely
  above 0. The mechanism doesn't follow the readers' gradient, which supports "one broad feature" or
  "the readers' mixing is in the examples, not the mechanism".
- "unclear": anything else (the two primary statistics disagree, or the intervals are too wide to tell
  the driver gradient from the reader's).
Known limit: if nearly every neuron's least pair is driver-unrelated, P1 is squeezed near a ceiling; P2's
continuous margin is there for that case.

Main confound and pre-registered controls.
- C1, facet size: the null threshold, and a pair's overlap, may depend on how diffuse (small, noisy) the
  facets are. Reported per quartile: median null threshold, facet size and number of facets. C1 is the
  margin's top-minus-rest difference (mean margin, S2 below) within facet-size strata: terciles of each
  neuron's median facet size over the selected neurons. Strata are combined with weights
  n_top * n_rest / (n_top + n_rest) and bootstrapped within quartile-by-stratum cells. Reading: if S2's
  interval excludes 0 and C1's has the same sign and excludes 0, facet size doesn't explain it; if S2's
  excludes 0 and C1's doesn't, facet size may explain it.
- C2, number of facets: a neuron with more facets has more pairs, so its least pair is more likely to be
  driver-unrelated, and strong neurons have fewer facets. C2 is P1 within strata of 2, 3, and 4 or more
  facets, combined and bootstrapped as C1. Read like C1, against P1.

Secondary (reported, no verdict): S1, the share of pairs driver-unrelated, top minus rest; S2, the mean
margin, top minus rest; S3, the reader's least-related pair driver-unrelated, top minus rest; and P1, P2,
S1 with head drivers.

v13 consistency check (no test): from data/v13's final checkpoint, v23's code must reproduce v13's
published end thresholds and final MLP overlaps; neurons in both selections must get the same pair
overlaps from both recordings (same facets, same background); and v13's 40 neurons get the primary measure
by peak quartile, as a description only (they were selected on the reader's flag).
"""

import numpy as np
from scipy.stats import spearmanr

from recgen.config import FINAL_STEP, SEED

MODEL = "410m"
DEPTH = "middle"
STEP = FINAL_STEP
PER_QUARTILE = 50
MIN_FACETS = 2
BOOTSTRAP = 1000
INTERVAL = (2.5, 97.5)
TOP = 3  # strongest peak quartile
FACET_COUNT_STRATA = (2, 3, 4)  # 4 means 4 or more

MORE_JOBS = "more jobs"
WEAK_SHARE_MORE = "weak neurons share more"
NO_GRADIENT = "no driver gradient"
UNCLEAR = "unclear"


def rng_for(purpose: int):
    return np.random.default_rng([SEED, 23, purpose])


def quartiles(values: np.ndarray) -> np.ndarray:
    """v12's quartiles: 0 weakest to 3 strongest, within the given set."""
    return np.digitize(values, np.quantile(values, [0.25, 0.5, 0.75]))


def select(quartile: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """PER_QUARTILE eligible sampled-neuron indices from each quartile (seed [SEED, 23, quartile]), sorted."""
    chosen = []
    for q in range(4):
        candidates = np.flatnonzero((quartile == q) & eligible)
        rng = np.random.default_rng([SEED, 23, q])
        chosen.extend(rng.choice(candidates, size=min(PER_QUARTILE, len(candidates)), replace=False))
    return np.sort(np.array(chosen, dtype=int))


def pair_order(key: str) -> tuple[int, int]:
    i, j = key.split("-")
    return int(i), int(j)


def least_overlapping(overlaps: dict[str, float]) -> str:
    return min(overlaps, key=lambda key: (overlaps[key], pair_order(key)))


def neuron_measures(overlaps: dict[str, float], threshold: float, reader_least: str) -> dict:
    """The per-neuron measures from its pair overlaps and its null threshold (v2: adjacent if above)."""
    least = least_overlapping(overlaps)
    return {"least_pair": least, "least_overlap": overlaps[least], "threshold": threshold,
            "least_unrelated": bool(overlaps[least] <= threshold),
            "share_unrelated": float(np.mean([o <= threshold for o in overlaps.values()])),
            "margin": float(overlaps[least] - threshold),
            "reader_least_unrelated": bool(overlaps[reader_least] <= threshold)}


def facet_size_strata(median_sizes: np.ndarray) -> np.ndarray:
    """Terciles (0, 1, 2) of each neuron's median facet size over the selected neurons."""
    return np.digitize(median_sizes, np.quantile(median_sizes, [1 / 3, 2 / 3]))


def facet_count_strata(n_facets: np.ndarray) -> np.ndarray:
    return np.minimum(n_facets, FACET_COUNT_STRATA[-1])


def _difference(quartile: np.ndarray, values: np.ndarray, strata: np.ndarray | None) -> float:
    if strata is None:
        return float(values[quartile == TOP].mean() - values[quartile < TOP].mean())
    total, weights = 0.0, 0.0
    for s in np.unique(strata):
        top, rest = values[(strata == s) & (quartile == TOP)], values[(strata == s) & (quartile < TOP)]
        if len(top) and len(rest):
            w = len(top) * len(rest) / (len(top) + len(rest))
            total += w * (top.mean() - rest.mean())
            weights += w
    return total / weights


def _resample(rng, cells: np.ndarray) -> np.ndarray:
    """Indices drawn with replacement within each cell."""
    idx = []
    for c in np.unique(cells):
        members = np.flatnonzero(cells == c)
        idx.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(idx)


def top_minus_rest(quartile: np.ndarray, values: np.ndarray, rng, strata: np.ndarray | None = None) -> dict:
    """Top quartile's mean minus the other three's (within strata and combined, if given), with a bootstrap
    interval resampling neurons within quartiles (and strata)."""
    values = np.asarray(values, float)
    cells = quartile * 100 + (0 if strata is None else strata)
    point = _difference(quartile, values, strata)
    draws = []
    for _ in range(BOOTSTRAP):
        idx = _resample(rng, cells)
        draws.append(_difference(quartile[idx], values[idx], None if strata is None else strata[idx]))
    return {"value": point, "interval": [float(x) for x in np.percentile(draws, INTERVAL)]}


def paired_difference(quartile: np.ndarray, a: np.ndarray, b: np.ndarray, rng) -> dict:
    """(top minus rest of a) minus (top minus rest of b) on the same neurons, with a paired bootstrap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    point = _difference(quartile, a, None) - _difference(quartile, b, None)
    draws = []
    for _ in range(BOOTSTRAP):
        idx = _resample(rng, quartile)
        draws.append(_difference(quartile[idx], a[idx], None) - _difference(quartile[idx], b[idx], None))
    return {"value": point, "interval": [float(x) for x in np.percentile(draws, INTERVAL)]}


def spearman(quartile: np.ndarray, peaks: np.ndarray, values: np.ndarray, rng) -> dict:
    peaks, values = np.asarray(peaks, float), np.asarray(values, float)
    point = float(spearmanr(peaks, values).statistic)
    draws = []
    for _ in range(BOOTSTRAP):
        idx = _resample(rng, quartile)
        draws.append(spearmanr(peaks[idx], values[idx]).statistic)
    return {"value": point, "interval": [float(x) for x in np.nanpercentile(draws, INTERVAL)]}


def reading(p1: list[float], p2: list[float], p3: list[float]) -> str:
    """The pre-registered reading from the intervals of P1, P2 and P3."""
    if p1[1] < 0 and p2[0] > 0:
        return MORE_JOBS
    if p1[0] > 0 and p2[1] < 0:
        return WEAK_SHARE_MORE
    if p1[1] >= 0 and p2[0] <= 0 and p3[0] > 0:
        return NO_GRADIENT
    return UNCLEAR


def control_reading(unstratified: list[float], stratified: list[float]) -> str:
    """C1 and C2: does the stratified difference keep the unstratified one's sign and exclude 0?"""
    if unstratified[0] <= 0 <= unstratified[1]:
        return "nothing to explain (unstratified interval includes 0)"
    same_side = (stratified[1] < 0) if unstratified[1] < 0 else (stratified[0] > 0)
    return "holds within strata" if same_side else "may be explained by the stratifying variable"
