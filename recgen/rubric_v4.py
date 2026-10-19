"""Pre-registered v4 rules, written and committed before they were computed on real data.

v2 judged facet adjacency by upstream drivers at the final checkpoint only. Recursive
generalization predicts that a later facet grows out of an earlier one's circuitry, so the
two should share drivers while the later one forms, even if they separate afterwards. v4
measures v2's driver overlaps at every v1 checkpoint and asks whether two facets shared
drivers during their formation. See README, "v4".

Formation window. A facet's gain at a checkpoint is the neuron's median pre-activation on the
facet's examples minus its median on background text. The facet forms from the first
checkpoint from which its gain stays at or above WINDOW_START of its final gain to the first
from which it stays at or above WINDOW_END (one dip allowed, as in v1). A pair's window spans
both facets' windows.

Adjacency at formation. For each checkpoint, a facet's drivers are found as in v2 (the
TOP_UNITS upstream MLP neurons whose direct contribution rises most over background). A pair
is adjacent at formation if its largest driver overlap within the pair's window exceeds the
NULL_PERCENTILE of the same statistic (largest overlap within that window) for null pairs:
each of the two facets against every facet of the other traced neurons, all read through
this neuron's weights. Taking the maximum over the window on both sides keeps the
false-positive rate at about 5% however many checkpoints the window spans.

Attention heads are a second, separate driver set: the TOP_HEADS of the 192 heads in layers 0
to 11 whose direct contribution rises most. Head adjacency at formation uses the same rule.
MLP-neuron drivers are the primary measure (comparable with v2); heads are reported beside them.

Labels are v1's and v2's (rubric_v2.classify) with adjacency at formation in place of
adjacency at the final checkpoint. A pair that is adjacent at formation but not at the final
checkpoint (v2) "branched then separated": the signature v2 could not see.
"""

import numpy as np

from recgen.config import CHECKPOINT_STEPS
from recgen.rubric import ALLOWED_DIPS
from recgen.rubric_v2 import NULL_PERCENTILE, TOP_UNITS  # noqa: F401  (re-exported: v4 keeps v2's values)

STEPS = CHECKPOINT_STEPS[1:]  # step 0 is random initialization: no drivers to compare
WINDOW_START = 0.25
WINDOW_END = 0.75
TOP_HEADS = 5  # of 192


def crossing_at(gains: np.ndarray, fraction: float) -> int | None:
    """First checkpoint from which the gain stays at or above `fraction` of its last value
    (allowing ALLOWED_DIPS dips), or None if the last value isn't positive."""
    if gains[-1] <= 0:
        return None
    above = gains >= fraction * gains[-1]
    for i in range(len(gains)):
        if above[i] and np.sum(~above[i + 1:]) <= ALLOWED_DIPS:
            return i
    return None


def formation_window(gains: np.ndarray) -> tuple[int, int] | None:
    """Checkpoint indices (first, last) over which the facet forms, or None if its final gain isn't positive."""
    start = crossing_at(gains, WINDOW_START)
    end = crossing_at(gains, WINDOW_END)
    if start is None or end is None:
        return None
    return start, max(start, end)


def pair_window(a: tuple[int, int] | None, b: tuple[int, int] | None) -> tuple[int, int] | None:
    if a is None or b is None:
        return None
    return min(a[0], b[0]), max(a[1], b[1])


def peak(series: np.ndarray, window: tuple[int, int]) -> float:
    """Largest value of a per-checkpoint series within the window (inclusive)."""
    return float(np.max(series[window[0]:window[1] + 1]))


def formation_threshold(null_series: list[np.ndarray], window: tuple[int, int]) -> float:
    return float(np.percentile([peak(s, window) for s in null_series], NULL_PERCENTILE))


def adjacent_at_formation(pair_series: np.ndarray, null_series: list[np.ndarray], window: tuple[int, int]) -> bool:
    return peak(pair_series, window) > formation_threshold(null_series, window)


def branched_then_separated(at_formation: bool, at_final: bool) -> bool:
    return at_formation and not at_final
