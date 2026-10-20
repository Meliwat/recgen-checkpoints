"""Post hoc, added after an independent review of v19; not pre-registered. A sensitivity analysis of the network
reader on GPT-2 for token position.

GPT-2 adds a learned absolute position embedding to its residual stream; Pythia applies rotary position
embeddings inside attention and adds none. In GPT-2's centered middle-layer residual, two random positions are
more alike the closer they are in their windows, which Pythia's v6 recordings don't show. Since the network
reader clusters a neuron's examples by cosine similarity against a null of random positions, position could make
examples look alike or unrelated for reasons that have nothing to do with what the neuron does.

This measures that, then reruns the network reader with a position subspace projected out of every vector
(examples and null), beside each unit's neuron directions, as analyze_v6.Layer does for the directions alone.

The position subspace of a layer is fixed by this rule: regress its stored residuals (every recorded example and
random position, centered on their mean) on POSITION_FEATURES of the position p / CONTEXT_LEN, and take the top
k right singular vectors of the fitted values. As a control, k random orthonormal directions are projected out
instead.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recgen.analyze_v6 import PRIMARY, Layer, NullSpace, detect, levels
from recgen.config import CONTEXT_LEN
from recgen.rubric_v5 import MODELS as PYTHIA_MODELS
from recgen.rubric_v5 import PRIMARY_DEPTH
from recgen.rubric_v6 import MERGE_PERCENTILE, embed, pair_similarities
from recgen.rubric_v19 import excess_over_models, h1
from recgen.sweep_v6 import v6_path

HARMONICS = (1, 2, 3)
POSITION_FEATURES = ("1", "p", "p^2", "log p") + tuple(f"{f}(2 pi {h} p)" for h in HARMONICS for f in ("sin", "cos"))
# name -> (which subspace, its dimension)
VARIANTS = {"position, 2 dims": ("position", 2), "position, 6 dims": ("position", 6), "random, 6 dims": ("random", 6)}
THREADS = 8


def position_features(positions: np.ndarray) -> np.ndarray:
    """POSITION_FEATURES of each position [n, len(POSITION_FEATURES)]."""
    p = np.asarray(positions, np.float64) / CONTEXT_LEN
    waves = [f(2 * np.pi * h * p) for h in HARMONICS for f in (np.sin, np.cos)]
    return np.stack([np.ones_like(p), p, p ** 2, np.log(p), *waves], axis=1)


def position_fit(z: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The regression of the centered residuals z [n, d] on position: the fitted values' right singular vectors
    [f, d], strongest first (the top k span the position subspace of dimension k), the share of the fit's squared
    norm the top 1, 2, ... of them capture, and the share of the residuals' squared norm the fit explains."""
    centered = z.astype(np.float64)
    centered -= centered.mean(axis=0)
    q, _ = np.linalg.qr(position_features(positions))
    fitted = q.T @ centered  # the least-squares fit is q @ fitted, which has fitted's singular values and vectors
    _, singular, vt = np.linalg.svd(fitted, full_matrices=False)
    explained = float(np.sum(fitted ** 2))
    return vt, np.cumsum(singular ** 2) / explained, explained / float(np.einsum("ij,ij->", centered, centered))


def random_subspace(d: int, k: int, rng) -> np.ndarray:
    """k random orthonormal directions [k, d]."""
    return np.linalg.qr(rng.standard_normal((d, k)))[0].T


def without(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """vectors [n, d] with the span of the orthonormal rows of basis [k, d] projected out."""
    return vectors - (vectors @ basis.T) @ basis


def null_position_correlation(null: np.ndarray, positions: np.ndarray) -> float:
    """Pearson correlation, over every pair of null vectors (already centered) [n, d], of their cosine similarity
    with how far apart their positions are."""
    units = null / np.linalg.norm(null, axis=1, keepdims=True)
    i, j = np.triu_indices(len(units), 1)
    gap = np.abs(positions[i].astype(np.float64) - positions[j])
    return float(np.corrcoef(pair_similarities(units), gap)[0, 1])


def recorded_null(path) -> tuple[np.ndarray, np.ndarray]:
    """A recording's random positions in v6's format: the residuals centered on their mean, and their positions."""
    data = np.load(path)
    random = data["random_rows"]
    null = data["z"][random].astype(np.float64)
    return null - null.mean(axis=0), data["keys"][random, 1]


class WithoutSubspace:
    """The network reader (analyze_v6.Layer's primary embedding) with a subspace [k, d] of orthonormal rows
    projected out of every example and null vector, as well as each unit's neuron directions. Removing the
    subspace and then what's left of the neuron directions removes the same span as both at once, so the null
    keeps one Gram matrix per layer, as in Layer."""

    def __init__(self, layer: Layer, basis: np.ndarray):
        mean, _, directions = layer.embeddings[PRIMARY]
        self.z, self.mean, self.basis = layer.z, mean, basis
        self.directions = without(directions, basis)
        self.null = NullSpace(without(layer.nulls[PRIMARY].vectors, basis))

    def flag(self, rows: np.ndarray, neurons: list[int]) -> bool:
        units = embed(self.z[rows], self.mean, np.vstack([self.directions[neurons], self.basis]))
        null_levels = levels(self.null.similarities(self.directions[neurons]))
        return bool(detect(units, null_levels["merge"][MERGE_PERCENTILE], null_levels["unrelated"])["unrelated_mixed"])


def flags_without(layer: Layer, basis: np.ndarray, pooled: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """The network reader's flag, with basis projected out, for every sampled neuron [S] and each pooled unit."""
    reader = WithoutSubspace(layer, basis)
    with ThreadPoolExecutor(THREADS) as pool:
        neurons = list(pool.map(lambda s: reader.flag(layer.rows[s], [s]), range(len(layer.neurons))))
        units = list(pool.map(lambda u: reader.flag(u["rows"], list(u["neurons"])), pooled))
    return np.array(neurons, float), np.array(units, float)


def model_sensitivity(layer: Layer, null_path, units: list[dict], rng) -> dict:
    """For one GPT-2 model: the null's position correlation, and for each of VARIANTS the subspace, the null's
    position correlation without it, and the network reader's flags without it for every sampled neuron and
    each 50/50 pool. A real unit's flag is its neuron's (analyze_v19.unit_rows checks this)."""
    null, null_positions = recorded_null(null_path)
    pooled = [u for u in units if u["group"] == "pooled 50/50"]
    vt, captured, r2 = position_fit(layer.z, layer.keys[:, 1])
    out = {"null_position_correlation": null_position_correlation(null, null_positions),
           "position_r2": r2, "variants": {}}
    for name, (kind, k) in VARIANTS.items():
        if kind == "position":
            basis, fit = vt[:k], {"captured_share_of_fit": float(captured[k - 1])}
        else:
            basis, fit = random_subspace(layer.z.shape[1], k, rng), {}
        neuron_flags, pooled_flags = flags_without(layer, basis, pooled)
        out["variants"][name] = {**fit, "null_position_correlation": null_position_correlation(without(null, basis), null_positions),
                                 "neuron_flags": neuron_flags, "pooled_flags": pooled_flags}
    return out


def summarize(by_model: dict[str, dict], neurons: dict[str, list[dict]], units: dict[str, list[dict]],
              h1_rng, excess_rng) -> dict:
    """Per variant, by model and pooled: how many network flags change, the flag rates, test (b)'s H1 and the
    network reader's test (a) excess. h1_rng and excess_rng make the generators of the original tests, so the
    bootstrap draws match them and only the flags differ."""
    out = {}
    for name in VARIANTS:
        per_model, groups, pairs, changed = {}, [], [], []
        for model, result in by_model.items():
            variant = result["variants"][name]
            new = variant["neuron_flags"]
            old = np.array([n["network"] for n in neurons[model]], float)
            real = [u["neurons"][0] for u in units[model] if u["group"] == "real"]
            pooled_old = np.array([u["network"] for u in units[model] if u["group"] == "pooled 50/50"], float)
            per_model[model] = {
                **{k: v for k, v in variant.items() if k not in ("neuron_flags", "pooled_flags")},
                "neurons_flagged": [float(old.mean()), float(new.mean())],
                "neuron_flags_unchanged": float(np.mean(old == new)),
                "real_flagged": [float(old[real].mean()), float(new[real].mean())],
                "pooled_50_50_flagged": [float(pooled_old.mean()), float(variant["pooled_flags"].mean())]}
            groups.append((np.array([n["log_peak"] for n in neurons[model]]), np.array([n["quartile"] for n in neurons[model]]), new))
            pairs.append((variant["pooled_flags"], new[real]))
            changed.append(old != new)
        excess = excess_over_models(pairs, excess_rng())
        out[name] = {"by_model": per_model, "neuron_flags_changed": float(np.mean(np.concatenate(changed))),
                     "h1": h1(groups, h1_rng()), "network_excess": {"excess": excess["excess"], "interval": excess["interval"]}}
    return out


def pythia_null_correlations() -> dict[str, float]:
    """The null's position correlation in Pythia's v6 recordings at the middle layer, as for GPT-2."""
    return {model: null_position_correlation(*recorded_null(v6_path(model, PRIMARY_DEPTH))) for model in PYTHIA_MODELS}
