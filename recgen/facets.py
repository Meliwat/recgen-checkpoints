"""A neuron's strongest examples, and the facets (clusters of related examples) they form.

Examples are embedded with an independent sentence model rather than the network
under study, so "related" means related to a reader, not by construction to the neuron.
"""

from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer

from recgen.activations import MIN_CONTEXT
from recgen.config import MODEL

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 50  # strongest examples per neuron, at most one per window
MIN_FACET_SIZE = 5
FACET_DISTANCE = 0.75  # average-linkage cosine distance at which clusters stop merging
CONTEXT_BEFORE = MIN_CONTEXT  # tokens of context shown and embedded before the activating token


@cache
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@cache
def embedder():
    return SentenceTransformer(EMBEDDER)


def top_examples(window_max: np.ndarray, window_pos: np.ndarray, neuron: int, k: int = TOP_K):
    """The neuron's k strongest windows as (window, position, activation), strongest first."""
    windows = np.argsort(-window_max[:, neuron])[:k]
    return [(int(w), int(window_pos[w, neuron]), float(window_max[w, neuron])) for w in windows]


def context_text(corpus: np.ndarray, window: int, pos: int) -> str:
    return tokenizer().decode(corpus[window, pos - CONTEXT_BEFORE:pos + 1])


def display_text(corpus: np.ndarray, window: int, pos: int, before: int = 24, after: int = 4) -> str:
    """Context with the activating token marked as [[token]]."""
    tok = tokenizer()
    left = tok.decode(corpus[window, max(0, pos - before):pos])
    token = tok.decode(corpus[window, pos:pos + 1])
    right = tok.decode(corpus[window, pos + 1:pos + 1 + after])
    return f"{left}[[{token}]]{right}".replace("\n", "⏎")


def embed(texts: list[str]) -> np.ndarray:
    return embedder().encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)


def find_facets(embeddings: np.ndarray) -> list[np.ndarray]:
    """Cluster example embeddings; return member indices for each cluster of at
    least MIN_FACET_SIZE, largest first. Smaller clusters are left out as misc."""
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=FACET_DISTANCE).fit_predict(embeddings)
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return sorted((c for c in clusters if len(c) >= MIN_FACET_SIZE), key=len, reverse=True)


def centroid(embeddings: np.ndarray) -> np.ndarray:
    mean = embeddings.mean(axis=0)
    return mean / np.linalg.norm(mean)
