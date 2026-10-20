"""Pre-registered v12 rules: a confirmatory test of v9's exploratory magnitude effect, written and
committed before looking at any of the held-out data below.

v9 found, after looking at the middle layer, that neurons with larger peak activations (a neuron's
largest per-window maximum over the corpus) are flagged "unrelated mixed" less often by the network
reader and by Haiku (top peak quartile about half the rate of the rest), and only faintly by MiniLM.
v12 tests that on data the finding didn't use, with controls for the obvious alternative
explanations.

Held-out data:
- network and MiniLM flags at the QUARTER and THREE-QUARTERS depths of all six sizes, already
  computed by v6 (network) and v5 (MiniLM) for every sampled neuron, never examined for magnitude;
- fresh Haiku readings at the middle layer: HAIKU_PER_QUARTILE sampled neurons from each peak
  quartile per model, none of them among v7's 100 real neurons, read with v7's prompt and CLI;
- if v11's Haiku readings of real neurons at the other depths exist when v12 is analyzed, they are
  added as further held-out Haiku data and reported separately.

Peak is log(peak / median peak of the layer's sampled neurons), so layers and sizes are comparable.

H1 (primary), per reader and depth: pooled over models, the flag rate of the top peak quartile minus
that of the other three quartiles is below zero, with a bootstrap interval (over neurons, within
models) whose high end is below zero; and the Spearman correlation of peak with the flag is
negative in at least MIN_MODELS of the six models.

H2 (controls): a logistic regression of the flag on standardized log peak, peakedness (peak over the
50th-ranked window maximum), density (share of windows at or above half the peak) and token purity
(share of the 50 strongest examples on the most common activating token), with a fixed effect per
model. H2 holds if the log-peak coefficient's bootstrap interval is entirely below zero. If H1
holds and H2 doesn't, the controls' coefficients say which one absorbs the effect.

Amendment, made after the first analysis run crashed and before any result was computed: a few
sampled neurons fire in fewer than 50 windows, so their 50th-ranked window maximum is 0 and
peakedness is undefined. They stay in H1 and are left out of H2, and their number is reported.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from recgen.config import SEED
from recgen.rubric_v5 import MODELS

HELD_OUT_DEPTHS = ("quarter", "three quarters")
HAIKU_DEPTH = "middle"
HAIKU_PER_QUARTILE = 25
MIN_MODELS = 5
BOOTSTRAP = 1000
COVARIATES = ("log_peak", "peakedness", "density", "token_purity")


def rng_for(model: str, purpose: int):
    return np.random.default_rng([SEED, 12, MODELS.index(model), purpose])


def neuron_stats(window_max: np.ndarray, top_tokens: list[int]) -> dict:
    """Peak, peakedness and density from one neuron's window maxima [W], and token purity from the
    activating tokens of its 50 strongest examples."""
    ordered = np.sort(window_max)[::-1]
    peak = float(ordered[0])
    fiftieth = float(ordered[49])
    return {"peak": peak, "peakedness": peak / fiftieth if fiftieth > 0 else float("nan"),
            "density": float(np.mean(window_max >= 0.5 * peak)),
            "token_purity": max(top_tokens.count(t) for t in set(top_tokens)) / len(top_tokens)}


def quartiles(values: np.ndarray) -> np.ndarray:
    """Quartile (0 weakest to 3 strongest) of each value within its own set."""
    return np.digitize(values, np.quantile(values, [0.25, 0.5, 0.75]))


def top_minus_rest(groups: list[tuple[np.ndarray, np.ndarray]], rng) -> tuple[float, tuple[float, float]]:
    """Pooled over models (each weighted by its neurons), the top-quartile flag rate minus the rest's,
    with a bootstrap interval resampling neurons within each model. groups: (quartile, flag) per model."""
    def statistic(sample):
        top = np.concatenate([f[q == 3] for q, f in sample])
        rest = np.concatenate([f[q < 3] for q, f in sample])
        return top.mean() - rest.mean()

    point = float(statistic(groups))
    draws = []
    for _ in range(BOOTSTRAP):
        sample = []
        for q, f in groups:
            idx = rng.integers(0, len(f), len(f))
            sample.append((q[idx], f[idx]))
        draws.append(statistic(sample))
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, (float(low), float(high))


def h1(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]], rng) -> dict:
    """groups: (log peak, quartile, flag) per model."""
    point, interval = top_minus_rest([(q, f) for _, q, f in groups], rng)
    rhos = [float(spearmanr(p, f).statistic) for p, _, f in groups]
    negative = sum(r < 0 for r in rhos)
    return {"top_minus_rest": point, "interval": list(interval), "spearman_by_model": rhos,
            "models_negative": negative, "holds": interval[1] < 0 and negative >= MIN_MODELS}


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std() if x.std() > 0 else x * 0


def design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Standardized covariates plus model indicators, and the flags."""
    covariates = np.column_stack([standardize(np.array([r[c] for r in rows], float)) for c in COVARIATES])
    models = sorted({r["model"] for r in rows})
    dummies = np.column_stack([[r["model"] == m for r in rows] for m in models[1:]]).astype(float) \
        if len(models) > 1 else np.zeros((len(rows), 0))
    return np.column_stack([covariates, dummies]), np.array([r["flag"] for r in rows], float)


def coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return LogisticRegression(C=np.inf, max_iter=2000).fit(x, y).coef_[0][:len(COVARIATES)]  # unpenalized


def h2(rows: list[dict], rng) -> dict:
    defined = [r for r in rows if all(np.isfinite(r[c]) for c in COVARIATES)]
    x, y = design(defined)
    point = coefficients(x, y)
    draws = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        draws.append(coefficients(x[idx], y[idx]))
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    out = {c: {"coefficient": float(p), "interval": [float(lo), float(hi)]}
           for c, p, lo, hi in zip(COVARIATES, point, low, high)}
    out["holds"] = bool(high[0] < 0)
    out["left_out_undefined_peakedness"] = len(rows) - len(defined)
    return out
