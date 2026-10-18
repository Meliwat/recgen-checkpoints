"""Pre-registered rules for classifying how a mixed-looking neuron's facets formed.

Written and committed before any trajectory was recorded. See README, "Rubric".
"""

from dataclasses import dataclass

import numpy as np

PRESENT = 0.5  # facet response, relative to the neuron's top-50 level at that checkpoint
ALLOWED_DIPS = 1  # checkpoints after emergence allowed to fall below PRESENT
MIN_SHARED_SUBSTRING = 2

UNSTABLE = "unstable"
MIXED_FROM_START = "mixed from the start"
PACKING_LIKE = "packing-like"
RECURSIVE_GENERALIZATION = "recursive generalization"
ONE_ORIGIN_UNRESOLVED = "one origin, timing unresolved"


@dataclass
class Facet:
    responses: np.ndarray  # response at each checkpoint, oldest first
    centroid: np.ndarray  # unit-norm sentence embedding of the facet's examples
    top_token: str  # most common activating token, normalized


def emergence(responses: np.ndarray) -> int | None:
    """Index of the first checkpoint from which the facet stays present, or None."""
    if responses[-1] < PRESENT:
        return None
    for i, r in enumerate(responses):
        if r >= PRESENT and np.sum(responses[i + 1:] < PRESENT) <= ALLOWED_DIPS:
            return i
    return None


def tokens_related(a: str, b: str) -> bool:
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= MIN_SHARED_SUBSTRING and shorter in longer


def adjacent(f: Facet, g: Facet, similarity_threshold: float) -> bool:
    return float(f.centroid @ g.centroid) >= similarity_threshold or tokens_related(f.top_token, g.top_token)


def connected(facets: list[Facet], similarity_threshold: float) -> bool:
    """Whether every facet can reach every other through a chain of adjacent facets."""
    reached, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for j in range(len(facets)):
            if j not in reached and adjacent(facets[i], facets[j], similarity_threshold):
                reached.add(j)
                frontier.append(j)
    return len(reached) == len(facets)


def classify(facets: list[Facet], similarity_threshold: float) -> str:
    times = [emergence(f.responses) for f in facets]
    if any(t is None for t in times):
        return UNSTABLE
    first = min(times)
    origin = [f for f, t in zip(facets, times) if t == first]
    if not connected(origin, similarity_threshold):
        return MIXED_FROM_START
    later = [(f, t) for f, t in zip(facets, times) if t > first]
    if not later:
        return ONE_ORIGIN_UNRESOLVED
    for f, t in later:
        earlier = [g for g, s in zip(facets, times) if s < t]
        if not any(adjacent(f, g, similarity_threshold) for g in earlier):
            return PACKING_LIKE
    return RECURSIVE_GENERALIZATION
