"""Pre-registered v5 rules, written and committed before any model was recorded.

v1 to v4 studied one layer of Pythia-410m. The essay expects bigger models to have narrower
neurons, so fewer mixed-looking ones. v5 runs v1's facet detector unchanged on six Pythia
sizes trained on the same data in the same order, and asks whether the share of mixed-looking
neurons falls with size. See README, "v5".

Neurons. For each model and depth, the layer is int(depth * layers). A neuron is near-dead if
its peak is below DEAD_FRACTION of the layer's median peak, and dense by v1's rule (half its
peak in over half the windows). From the rest, SAMPLE_SIZE are drawn with a fixed seed (all of
them if fewer).

Facets are v1's (recgen.facets): the neuron's 50 strongest windows, one example each, embedded
with the sentence model and clustered; clusters of 5 or more are facets.

Primary measure: the share of sampled neurons that are "unrelated mixed": 2 or more facets,
with the least-related pair's centroid similarity below UNRELATED_SIMILARITY. Secondary: the
share with 2 or more facets of any kind (v1's selection criterion). Middle depth is primary;
the other depths are reported beside it.

Decision, on the primary measure at middle depth: "fits the essay" if the share falls with
size (Spearman correlation with size rank at most -MIN_RHO) and the largest model's bootstrap
interval lies entirely below the smallest's; "against the essay" if it rises by the same
standard; otherwise "no clear trend".
"""

import numpy as np
from scipy.stats import spearmanr

MODELS = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]  # smallest first; 6.9b doesn't fit in 16 GB
DEPTHS = {"quarter": 0.25, "middle": 0.5, "three quarters": 0.75}
PRIMARY_DEPTH = "middle"
DEAD_FRACTION = 0.5
DENSE_STRONG_FRACTION = 0.5  # v1: a window is strong at half the neuron's peak...
DENSE_MAX_WINDOWS = 0.5  # ...and a neuron strong in over half the windows is dense
SAMPLE_SIZE = 2048  # the smallest model's layer width
UNRELATED_SIMILARITY = 0.1
BOOTSTRAP = 1000
INTERVAL = 95
MIN_RHO = 0.8

FITS = "fits the essay"
AGAINST = "against the essay"
NO_TREND = "no clear trend"


def layer_index(n_layers: int, depth: float) -> int:
    return int(depth * n_layers)


def eligible(window_max: np.ndarray) -> np.ndarray:
    """Indices of neurons that are neither near-dead nor dense, from window maxima [W, N]."""
    peak = window_max.max(axis=0)
    strong_share = (window_max > DENSE_STRONG_FRACTION * peak).mean(axis=0)
    return np.flatnonzero((peak >= DEAD_FRACTION * np.median(peak)) & (strong_share <= DENSE_MAX_WINDOWS))


def sample(neurons: np.ndarray, rng) -> np.ndarray:
    if len(neurons) <= SAMPLE_SIZE:
        return np.sort(neurons)
    return np.sort(rng.choice(neurons, size=SAMPLE_SIZE, replace=False))


def unrelated_mixed(n_facets: int, least_related_similarity: float | None) -> bool:
    return n_facets >= 2 and least_related_similarity < UNRELATED_SIMILARITY


def bootstrap_interval(flags: np.ndarray, rng) -> tuple[float, float]:
    """Percentile interval of the share of True flags, resampling neurons."""
    shares = rng.choice(flags, size=(BOOTSTRAP, len(flags)), replace=True).mean(axis=1)
    tail = (100 - INTERVAL) / 2
    return float(np.percentile(shares, tail)), float(np.percentile(shares, 100 - tail))


def decide(shares: list[float], intervals: list[tuple[float, float]]) -> tuple[str, float]:
    """Shares and intervals ordered smallest model first."""
    rho = float(spearmanr(np.arange(len(shares)), shares).statistic)
    smallest, largest = intervals[0], intervals[-1]
    if rho <= -MIN_RHO and largest[1] < smallest[0]:
        return FITS, rho
    if rho >= MIN_RHO and largest[0] > smallest[1]:
        return AGAINST, rho
    return NO_TREND, rho
