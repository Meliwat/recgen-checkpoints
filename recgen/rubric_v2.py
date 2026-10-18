"""Pre-registered v2 rules, written and committed before they were computed on real data.

v1 judged facet adjacency with the same sentence model that selected neurons for having
unrelated facets, so a two-facet neuron could not be labeled recursive generalization.
v2 keeps v1's emergence times and labels and judges adjacency with the network's own signal:
which upstream neurons drive this neuron on each facet. It also adds a bridge search. See
README, "v2".

Adjacency. A facet's drivers are the TOP_UNITS upstream MLP neurons (layers 0 to 11) whose
direct contribution to this neuron's pre-activation rises most on the facet's examples,
relative to random text. Two facets are adjacent if their drivers overlap more than the
NULL_PERCENTILE of the neuron's null: overlaps between its own facets and the facets of the
other traced neurons, all read through this neuron's weights, so shared readout weights
can't make two facets look adjacent.

Bridge search. For a later facet B and a facet A that emerged before it, "bridge" text is
the BRIDGE_SIZE pool texts most similar to both (by the smaller of their sentence-model
similarities to A's and B's centroids). Its B-ward gain is the neuron's median pre-activation
on the bridge minus that on a control matched to the bridge on similarity to A but not
similar to B ("not similar": at or below the pool's median similarity). B's own gain is its
examples' median pre-activation minus the pool's. Gains are measured at every recorded
checkpoint from A's emergence to B's: v1's checkpoints plus DENSE_PER_INTERVAL more per
interval. If the neuron's tuning spread from A through intermediate text, the bridge's gain
should reach half its level well before B's does. If B was attached directly, both are
driven by the same B-ward weights and rise together. A placebo gets the same test: pool text
matched to the bridge on similarity to B but not similar to A (gain: its median minus the
pool's), which should rise with B either way.
"""

import numpy as np

from recgen.rubric import (
    ALLOWED_DIPS, MIXED_FROM_START, ONE_ORIGIN_UNRESOLVED, PACKING_LIKE, RECURSIVE_GENERALIZATION, UNSTABLE,
)

TOP_UNITS = 100  # of 49,152 upstream MLP neurons
NULL_PERCENTILE = 95
# Reported next to the primary result, never replacing it.
SENSITIVITY_TOP_UNITS = (25, 50, 200)
SENSITIVITY_NULL_PERCENTILE = 75  # v1's percentile

BRIDGE_SIZE = 50  # pool texts per bridge and per control, at most one per window
DENSE_PER_INTERVAL = 3  # extra checkpoints between consecutive v1 checkpoints, on Pythia's 1,000-step grid
PYTHIA_STEP_GRID = 1000
HALF = 0.5
MIN_BRIDGE_GAIN = 0.2  # bridge's gain at B's emergence, as a fraction of B's own gain then
MIN_LEAD = 2  # checkpoints by which the bridge must cross half its level before B does

BRIDGE_LEADS = "bridge leads"
BRIDGE_WITH_OR_AFTER = "bridge rises with or after the facet"
NO_BRIDGE_RESPONSE = "no bridge response"
WINDOW_TOO_SHORT = "window too short"


def drivers(contributions: np.ndarray, k: int = TOP_UNITS) -> np.ndarray:
    """The k upstream units with the largest positive contributions (fewer if fewer are positive)."""
    order = np.argsort(-contributions, kind="stable")[:k]
    return order[contributions[order] > 0]


def overlap(a: np.ndarray, b: np.ndarray, k: int = TOP_UNITS) -> float:
    return len(np.intersect1d(a, b)) / k


def null_threshold(null_overlaps, percentile: float = NULL_PERCENTILE) -> float:
    return float(np.percentile(null_overlaps, percentile))


def adjacency(overlaps: np.ndarray, threshold: float) -> np.ndarray:
    """Facets i and j are adjacent if their drivers overlap more than the null threshold."""
    return overlaps > threshold


def connected(adjacent: np.ndarray, members: list[int]) -> bool:
    """Whether every member can reach every other through a chain of adjacent members."""
    reached, frontier = {members[0]}, [members[0]]
    while frontier:
        i = frontier.pop()
        for j in members:
            if j not in reached and adjacent[i, j]:
                reached.add(j)
                frontier.append(j)
    return len(reached) == len(members)


def classify(times: list[int | None], adjacent: np.ndarray) -> str:
    """v1's labels and order of checks, with adjacency given as a matrix.
    times are emergence checkpoint indices (None: never emerges)."""
    if any(t is None for t in times):
        return UNSTABLE
    first = min(times)
    if not connected(adjacent, [i for i, t in enumerate(times) if t == first]):
        return MIXED_FROM_START
    later = [i for i, t in enumerate(times) if t > first]
    if not later:
        return ONE_ORIGIN_UNRESOLVED
    for i in later:
        if not any(adjacent[i, j] for j, t in enumerate(times) if t < times[i]):
            return PACKING_LIKE
    return RECURSIVE_GENERALIZATION


def dense_grid(grid: list[int], per_interval: int = DENSE_PER_INTERVAL) -> list[int]:
    """The v1 checkpoints plus up to per_interval evenly spaced ones between each consecutive
    pair, rounded to Pythia's 1,000-step grid (none fit between 1k and 2k, for example)."""
    steps = set(grid)
    for a, b in zip(grid, grid[1:]):
        if a < PYTHIA_STEP_GRID:
            continue
        for x in np.linspace(a, b, per_interval + 2)[1:-1]:
            step = int(round(x / PYTHIA_STEP_GRID)) * PYTHIA_STEP_GRID
            if a < step < b:
                steps.add(step)
    return sorted(steps)


def crossing(gains: np.ndarray) -> int | None:
    """First checkpoint from which the gain stays at or above HALF its last value (allowing
    ALLOWED_DIPS dips), or None if the last value isn't positive."""
    if gains[-1] <= 0:
        return None
    above = gains >= HALF * gains[-1]
    for i in range(len(gains)):
        if above[i] and np.sum(~above[i + 1:]) <= ALLOWED_DIPS:
            return i
    return None


def bridge_lead(facet_gains: np.ndarray, bridge_gains: np.ndarray) -> int | None:
    """Checkpoints by which the bridge crosses half its level before the facet does (negative:
    after). Gains run from A's emergence to B's. None if the bridge barely responds by then."""
    if facet_gains[-1] <= 0 or bridge_gains[-1] < MIN_BRIDGE_GAIN * facet_gains[-1]:
        return None
    return crossing(facet_gains) - crossing(bridge_gains)


def bridge_outcome(facet_gains: np.ndarray, bridge_gains: np.ndarray) -> str:
    if len(facet_gains) < MIN_LEAD + 1:
        return WINDOW_TOO_SHORT
    lead = bridge_lead(facet_gains, bridge_gains)
    if lead is None:
        return NO_BRIDGE_RESPONSE
    return BRIDGE_LEADS if lead >= MIN_LEAD else BRIDGE_WITH_OR_AFTER
