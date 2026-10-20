"""Pre-registered v9 rules, written and committed before any v9 reading.

v7's positive control pooled two neurons' strongest examples by fixed counts (25 + 25, 35 + 15,
45 + 5). That is the easiest kind of mixing to detect: both halves are at full strength, and pools
can repeat a text window, which real neurons can't. v9 builds a more natural synthetic
polysemantic neuron: the union of two real neurons. Its activation in a window is the larger of
the two neurons' activations there, each divided by that neuron's scale, and its 50 strongest
examples come from its own ranking, one per window. How many examples each neuron contributes is
then set by their activation distributions, as it would be for a real neuron with two jobs.

Because the union takes a maximum, its top 50 windows are always among the two neurons' own top
50 windows (any window above one of them in the union's ranking is also above it in that neuron's
own ranking). So every example is already embedded for MiniLM (v5) and recorded for the network
reader (v6), and only the language model needs new readings. A pair whose union examples aren't all
found among the two neurons' recorded examples (possible only through tied activations) is replaced
by the next seeded pair.

Units, per model (middle layer): N_UNION unions of two random sampled neurons, on their RAW
activations (no rescaling: a unit that does both jobs at each neuron's own magnitude). Real neurons
are v7's 100, with v7's Haiku readings reused (same units, same prompt). Readers are v7's three,
unchanged.

Amendment, made after building the units and before any reading or reader output: the first
version of these rules (commit bace9af) scaled each neuron by its TOP_50 level. That makes the two
neurons cross at about their 25th example, so every union came out 25/25 (checked on all six
models), which defeats the aim of natural proportions. RAW gives minority counts from 0 to 25
(median 2 to 7 by model), so it is now primary; PEAK and TOP_50 are reported beside it for
MiniLM and the network reader only. Nothing else changed.

Pre-registered tests, per reader:
1. Detection: the union-minus-real flag rate, averaged over the six models, with a bootstrap
   interval over units. The reader detects natural unions if the interval's low end is above 0.
2. Against v7's pools: the same average for v7's 50/50 pools minus that for unions, with a
   bootstrap interval. Natural unions are expected to be harder, so this is expected to be positive.
3. Dose-response: the flag rate of unions by how many of the 50 examples come from the minority
   neuron (MINORITY_BINS), pooled over models; the top bin minus the bottom bin, with an interval.
4. Retest: RETEST_PER_MODEL unions per model read by Haiku a second time.
Reported beside these and not used in any test: the same pairs scaled by each neuron's PEAK and
by its TOP_50 level, read by MiniLM and the network reader only.
"""

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import MODELS

N_UNION = 150
TOP_K = 50
RAW, TOP_50, PEAK = "raw", "top-50 level", "peak"
PRIMARY = RAW
SENSITIVITY = (PEAK, TOP_50)
MINORITY_BINS = [(0, 5), (5, 15), (15, 26)]  # minority examples out of 50: [low, high)
RETEST_PER_MODEL = 10
BOOTSTRAP = 2000


def rng_for(model: str, purpose: int):
    return np.random.default_rng([SEED, 9, MODELS.index(model), purpose])


def scale(window_max: np.ndarray, how: str) -> float:
    """A neuron's scale from its window maxima [W]."""
    if how == RAW:
        return 1.0
    if how == PEAK:
        return float(window_max.max())
    return float(np.median(np.sort(window_max)[-TOP_K:]))


def union_top(max_a: np.ndarray, pos_a: np.ndarray, max_b: np.ndarray, pos_b: np.ndarray,
              scale_a: float, scale_b: float, k: int = TOP_K) -> list[tuple[int, int, int]]:
    """The union neuron's k strongest windows as (window, position, source 0 or 1), strongest first."""
    a, b = max_a.astype(np.float64) / scale_a, max_b.astype(np.float64) / scale_b
    union = np.maximum(a, b)
    windows = np.argsort(-union, kind="stable")[:k]
    return [(int(w), int(pos_a[w] if a[w] >= b[w] else pos_b[w]), 0 if a[w] >= b[w] else 1) for w in windows]


def minority_count(sources: list[int]) -> int:
    return min(sources.count(0), sources.count(1))


def minority_bin(count: int) -> int:
    return next(i for i, (low, high) in enumerate(MINORITY_BINS) if low <= count < high)


def mean_excess(groups: list[tuple[np.ndarray, np.ndarray]], rng) -> tuple[float, tuple[float, float]]:
    """Mean over models of (first rate - second rate), with a bootstrap interval resampling units
    within each model. groups: one (first flags, second flags) pair per model."""
    point = float(np.mean([x.mean() - y.mean() for x, y in groups]))
    draws = [np.mean([rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for x, y in groups])
             for _ in range(BOOTSTRAP)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, (float(low), float(high))
