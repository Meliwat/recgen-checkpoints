"""Pre-registered v6 rules, written and committed before any v6 data was recorded.

v5 ran v1's facet detector, which groups a neuron's strongest examples with a sentence model
(MiniLM), on six Pythia sizes. Its review found that this detector behaves differently with
size, splits one pattern by genre or language, and that the middle-layer decline beyond 70m
depends on its settings. v6 reruns v5's sweep with a detector that reads the network's own
representation of each example. See README, "v6".

Embedding. An example (window, position) of a neuron in layer L is the residual stream entering
block L at that position (the input to the MLP's LayerNorm), LayerNorm-normalized (centered and
divided by its standard deviation, without the gain and bias), minus the mean of the same vector
over N_RANDOM random corpus positions, with the component along the neuron's folded input
weight direction (its input weights times the LayerNorm gain) projected out. The neuron's
pre-activation is that direction's dot product with the normalized vector plus a constant, so
without the projection its strongest examples would look alike by construction.

Calibration. A cosine similarity doesn't mean the same thing in MiniLM's space and in each
model's residual stream, so v6 matches v1's operating point through a null: the similarities
between all pairs of the N_RANDOM random positions. In MiniLM's null on these positions, v1's
merge level (similarity 0.25, cosine distance 0.75) is the MERGE_PERCENTILE-th percentile and
v5's unrelated cutoff (0.1) the UNRELATED_PERCENTILE-th. For each neuron, the random positions
are embedded as its examples are (its direction projected out), and its thresholds are those
percentiles of their pairwise similarities: average-linkage clusters merge while their average
similarity is above the merge level, and two facets are unrelated if their centroids are less
similar than the unrelated level. The percentiles come from MiniLM alone and were computed
before any residual was recorded; analyze_v6 recomputes them.

Unchanged from v5: models, depths, eligibility, the sampled neurons (same seeds), the top 50
examples, the minimum facet size, the primary (unrelated mixed at middle depth) and secondary
(2+ facets) measures, bootstrap intervals, and decide().

Comparison with v5:
- Per-neuron agreement at each layer: of the unrelated-mixed flags (share agreeing and Cohen's
  kappa), and of how the two detectors partition each neuron's top 50 (adjusted Rand index;
  examples outside facets count as singletons).
- v5's verdict at a depth and measure holds if v6's verdict there is the same.
- Verdicts without 70m are secondary: most of v5's middle-layer drop is between 70m and 160m.

Detector checks, for both detectors, reported beside the verdicts but not part of them:
- The share of neurons with no facet, and the median similarity between a neuron's top examples
  as a percentile of its null. v5's review found both change with size under MiniLM.
- What each detector groups by: token purity, the share of a facet's examples on its most common
  activating token (median over facets).
- Power: N_MIXTURES synthetic neurons per layer, each the MIXTURE_PART strongest examples of two
  sampled neurons drawn at random (both directions projected out, in the null too). The share
  flagged unrelated mixed is the detector's power on a two-neuron mixture. If it falls with size
  (decide() gives FITS), a falling real share can't be told apart from lost power.

Sensitivity, reported beside the verdicts: v5's other detector settings, mapped through MiniLM's
null (SENSITIVITY_SETTINGS), and three embedding variants: with the LayerNorm gain (the vector
the MLP actually reads, with the neuron's raw input weights projected out), uncentered, and
unprojected.

Pilot. Before this file was committed, the recorder and detector were tried on pythia-14m and
31m, which aren't in the sweep. Activations rebuilt from the stored residuals matched within
0.01, and the detector wasn't degenerate: it found 2+ facets in about 80% of neurons (MiniLM: 41
to 45%), split mostly by activating token. Token purity was added as a diagnostic after that;
no threshold or rule was changed.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from recgen.activations import MIN_CONTEXT
from recgen.config import CONTEXT_LEN, SEED
from recgen.facets import FACET_DISTANCE, MIN_FACET_SIZE
from recgen.rubric_v5 import UNRELATED_SIMILARITY

N_RANDOM = 1024
RANDOM_POSITIONS_SEED = 6
# MiniLM similarity levels and the percentiles of MiniLM's random-pair null they sit at, computed
# on random_positions() before any v6 data: 0.25 is v1's merge level, 0.1 v5's unrelated cutoff,
# 0.3 and 0.2 the merge levels of v5's sensitivity distances 0.7 and 0.8.
MINILM_PERCENTILES = {0.3: 99.2, 1 - FACET_DISTANCE: 98.2, 0.2: 96.0, UNRELATED_SIMILARITY: 81.0}
MERGE_PERCENTILE = MINILM_PERCENTILES[1 - FACET_DISTANCE]
UNRELATED_PERCENTILE = MINILM_PERCENTILES[UNRELATED_SIMILARITY]
# v5's sensitivity settings, (cosine distance, minimum facet size) -> (merge percentile, minimum facet size)
SENSITIVITY_SETTINGS = {(0.7, 5): (99.2, 5), (0.8, 5): (96.0, 5), (0.75, 4): (98.2, 4), (0.75, 6): (98.2, 6)}
EMBEDDING_VARIANTS = ("with LayerNorm gain", "uncentered", "unprojected")
N_MIXTURES = 500
MIXTURE_PART = 25


def random_positions(n_windows: int) -> np.ndarray:
    """N_RANDOM (window, position) pairs [N_RANDOM, 2]: one position with full context in each of
    N_RANDOM random windows, sorted by window. The same text for every model."""
    rng = np.random.default_rng([SEED, RANDOM_POSITIONS_SEED])
    windows = np.sort(rng.choice(n_windows, size=N_RANDOM, replace=False))
    return np.stack([windows, rng.integers(MIN_CONTEXT, CONTEXT_LEN, size=N_RANDOM)], axis=1)


def embed(z: np.ndarray, mean: np.ndarray | None, directions: np.ndarray | None,
          scale: np.ndarray | None = None) -> np.ndarray:
    """Unit vectors from normalized residuals z [E, d]: minus mean (if given), times scale (if
    given), with the span of directions [k, d] (if given) projected out."""
    x = z.astype(np.float64) - (0 if mean is None else mean)
    if scale is not None:
        x = x * scale
    if directions is not None:
        basis, _ = np.linalg.qr(np.atleast_2d(directions).astype(np.float64).T)
        x = x - (x @ basis) @ basis.T
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pair_similarities(units: np.ndarray) -> np.ndarray:
    """Cosine similarity of every pair of distinct rows of unit vectors."""
    return (units @ units.T)[np.triu_indices(len(units), 1)]


def thresholds(null: np.ndarray, merge_percentile: float = MERGE_PERCENTILE) -> tuple[float, float]:
    """(merge level, unrelated level) from a neuron's null similarities."""
    merge, unrelated = np.percentile(null, [merge_percentile, UNRELATED_PERCENTILE])
    return float(merge), float(unrelated)


def percentile_rank(value: float, null: np.ndarray) -> float:
    """Percentage of null similarities below value."""
    return 100 * float(np.mean(null < value))


def find_facets(units: np.ndarray, merge_similarity: float, min_size: int = MIN_FACET_SIZE) -> list[np.ndarray]:
    """v1's average-linkage clustering, merging while the average similarity is above merge_similarity.
    Member indices of each cluster of at least min_size, largest first."""
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - merge_similarity).fit_predict(units)
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return sorted((c for c in clusters if len(c) >= min_size), key=len, reverse=True)


def least_related(units: np.ndarray, facets: list[np.ndarray]) -> tuple[float | None, tuple[int, int] | None]:
    """The lowest centroid similarity between two facets, and which pair, as in v1."""
    if len(facets) < 2:
        return None, None
    centroids = [units[f].mean(axis=0) / np.linalg.norm(units[f].mean(axis=0)) for f in facets]
    return min((float(centroids[i] @ centroids[j]), (i, j))
               for i in range(len(facets)) for j in range(i + 1, len(facets)))


def unrelated_mixed(n_facets: int, least_related_similarity: float | None, unrelated_similarity: float) -> bool:
    return n_facets >= 2 and least_related_similarity < unrelated_similarity


def partition(facets: list[np.ndarray], n_examples: int) -> np.ndarray:
    """A label per example: its facet's index, or a label of its own if it's in no facet."""
    labels = np.arange(len(facets), len(facets) + n_examples)
    for i, facet in enumerate(facets):
        labels[facet] = i
    return labels


def token_purity(tokens: list[int]) -> float:
    """Share of a facet's examples on its most common activating token."""
    return max(tokens.count(t) for t in set(tokens)) / len(tokens)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement of two boolean flag arrays beyond what their rates give by chance."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    observed = np.mean(a == b)
    chance = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((observed - chance) / (1 - chance)) if chance < 1 else float("nan")
