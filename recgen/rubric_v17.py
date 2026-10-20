"""Pre-registered v17 rules, written and committed before any v17 recording or reading.

Question. v12 confirmed, on held-out data, that neurons with a larger peak activation are flagged
"unrelated mixed" much less by the network reader and by Haiku, and that most of this is shared with
coherence: how alike a neuron's 50 strongest examples are in the network's representation. Its
review's main objection: the flag and coherence are properties of the top-50 sample, not of the
neuron, and a weak neuron's top 50 sits at a different fraction of its own range (the median
50th-strongest window is at 0.76 to 0.80 of the peak in the weakest peak quartile, 0.64 to 0.76 in
the strongest). v17 reads examples from a fixed band of each neuron's own range, at the middle
layer (DEPTH) of all six sizes, and asks:
1. Within a neuron: do examples below the top 50 look more mixed than the top 50? This is the
   toy-model-of-superposition prediction that a neuron's largest activations are clean and its
   smaller ones more polysemantic (Elhage et al. 2022).
2. Across neurons: at the same fraction of each neuron's own peak, do weakly firing neurons still
   look more mixed and less coherent than strongly firing ones?

Band. A neuron's band windows are those whose window maximum (v5's data["max"]) is at least
BAND[0] and below BAND[1] times its peak (its largest window maximum), leaving out the neuron's own
top-50 windows (v6's recorded examples), so the band and top-50 samples never share a window. Per
neuron, BAND_SIZE band windows are drawn uniformly without replacement with
band_rng(model, neuron id), one example per window at that window's strongest position (v5's
data["pos"]). Neurons with fewer than BAND_SIZE band windows are excluded from everything below,
in both halves of every pair, and counted by model and peak quartile.

Before this file was committed, band window counts were computed from v5's window maxima only (no
residual was recorded, no detector run, no reading made): with the top-50 windows left out, 27 of
the 11,801 sampled neurons (1 to 11 per model, all in the strongest peak quartile) and 2 of v12's
600 fresh neurons have fewer than 50 band windows. The top 50 overlapped the band in 12 to 77
neurons per model, which is why those windows are left out. The band is much wider for weak
neurons: the median neuron of the weakest peak quartile has 44 to 53% of all windows in its band,
the strongest quartile 15 to 25%.

Readers.
- network: v6's detector unchanged (recgen.analyze_v6), on band examples. The residual stream
  entering the layer at the band positions is recorded LayerNorm-normalized
  (recgen.sweep_v6.normalized_residuals) and stored in data/v17/. It is centered with v6's
  random-position mean, the neuron's folded input direction is projected out, and v6's null for
  that neuron (its random positions in data/v6) gives v6's merge and unrelated levels. Per neuron:
  band flag (unrelated mixed), band n_facets, and band coherence (v6's top_pair_percentile: the
  median pairwise similarity of the 50 examples as a percentile of the neuron's null). The same
  code recomputes each neuron's top-50 values from v6's recordings, and they must equal v6's
  published flags. Neurons: all of v6's sampled neurons at DEPTH.
- Haiku: v12's 600 fresh neurons (recgen.read_v12.fresh_units), each shown as its 50 band examples
  exactly as v7 shows a unit (recgen.llm_reader.unit_prompt and read: same prompt, CLI, marked
  token, no activations, order seeded by the unit id), so each has a paired top-50 reading (v12's)
  and band reading (v17's). Cached in data/v17/llm/. RETEST of them, drawn with rng_for, are read a
  second time (data/v17/llm_retest/). Drift check: the same RETEST neurons' top-50 prompts,
  byte-identical to v12's, are read again (data/v17/llm_top_reread/), to see whether Haiku's
  labels have moved since v12. Reading stops if the CLI-reported cost of v17's readings passes
  SPEND_LIMIT_USD.

Tests. Each runs per reader and measure: network flag, network coherence, Haiku flag. Flags are
1 for unrelated mixed and 0 otherwise; coherence is in percentile points. MIXED_SIGN orients each
measure so that higher means more mixed: +1 for flags, -1 for coherence.
- T1, within neuron: each neuron's band value minus its top-50 value; each model's mean difference,
  averaged over the six models with equal weight, with a 95% bootstrap interval resampling neurons
  within models (BOOTSTRAP draws). "band looks more mixed" if the interval lies entirely on the
  mixed side (flag above 0, coherence below 0), "band looks less mixed" if entirely on the other
  side, otherwise "no clear difference".
- T2, the magnitude effect in the band: v12's H1 (recgen.rubric_v12.h1) on band values oriented by
  MIXED_SIGN. That is, the top peak quartile minus the other three, pooled over models, with peak
  quartiles taken among all sampled neurons of the layer (for Haiku, v12's quartile of each fresh
  neuron); a bootstrap interval resampling neurons within models; and the Spearman correlation of
  log peak with the value negative in at least v12's MIN_MODELS (5) of the six models. For coherence this is
  sign reversed: it holds if strong neurons are more coherent. T2 also gives the same statistic on
  the same neurons' top-50 values, and the paired difference band minus top: v12's top_minus_rest
  applied to each neuron's oriented band-minus-top difference, which is exactly the difference of
  the two statistics under a paired bootstrap.

Verdict, per reader and measure: PERSISTS ("persists below the top") if T2 holds in the band;
TOP50_SPECIFIC ("top-50 specific") if the band statistic's interval includes 0 and the paired
band-minus-top interval excludes 0; otherwise UNCLEAR.

Caveat, stated before any result: band examples of dense neurons approach random positions, since
up to half of all windows fall in a weak neuron's band. Diffuse examples form fewer facets of at
least 5, and the network reader's flag needs two, so its flag rate may fall in the band for
everyone even where the examples are more varied. Coherence is co-primary for that reason. As a
reference (not a test), the network detector is also run per neuron on BAND_SIZE of v6's random
positions drawn with reference_rng, to show where random text sits on both measures.

Secondary, pre-registered but not part of the verdict: whether band density explains a band effect.
- S1: v12's H2 (recgen.rubric_v12.h2) on band flags, for both readers, with v12's covariates
  (standardized log peak, peakedness, density, token purity), where token purity is that of the
  band examples.
- S2: the same covariates and model fixed effects in a least-squares regression of oriented band
  coherence, with a bootstrap interval for log peak's coefficient (ols_coefficients).

Not run: the optional secondary band [0.20, 0.40). It would add about 1.8 GB of stored residuals,
and the disk had 17 GB free while other agents were downloading models.
"""

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import MODELS
from recgen.rubric_v12 import BOOTSTRAP, COVARIATES, design, h1, top_minus_rest

DEPTH = "middle"
BAND = (0.40, 0.60)  # fractions of the neuron's peak: at least BAND[0], below BAND[1]
BAND_SIZE = 50
BAND_SEED = 17
RETEST = 30  # 5 per model
SPEND_LIMIT_USD = 25.0
MIXED_SIGN = {"flag": 1, "coherence": -1}

MORE_MIXED, LESS_MIXED, NO_DIFFERENCE = "band looks more mixed", "band looks less mixed", "no clear difference"
PERSISTS, TOP50_SPECIFIC, UNCLEAR = "persists below the top", "top-50 specific", "unclear"


def band_rng(model: str, neuron_id: int):
    """The generator that draws one neuron's band windows."""
    return np.random.default_rng([SEED, BAND_SEED, 0, MODELS.index(model), neuron_id])


def reference_rng(model: str, neuron_id: int):
    """The generator that draws one neuron's random-position reference sample."""
    return np.random.default_rng([SEED, BAND_SEED, 1, MODELS.index(model), neuron_id])


def rng_for(model: str, purpose: int):
    return np.random.default_rng([SEED, BAND_SEED, 2, MODELS.index(model), purpose])


def t1(pairs: list[tuple[np.ndarray, np.ndarray]], sign: int, rng) -> dict:
    """pairs: (band, top) values per model, paired by neuron. sign: MIXED_SIGN of the measure."""
    differences = [np.asarray(band, float) - np.asarray(top, float) for band, top in pairs]
    by_model = [float(d.mean()) for d in differences]
    draws = [np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in differences]) for _ in range(BOOTSTRAP)]
    low, high = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    mixed_low, mixed_high = sorted((sign * low, sign * high))
    reading = MORE_MIXED if mixed_low > 0 else LESS_MIXED if mixed_high < 0 else NO_DIFFERENCE
    return {"band_minus_top": float(np.mean(by_model)), "interval": [low, high], "by_model": by_model, "reading": reading}


def verdict(band: dict, difference_interval: list[float]) -> str:
    """band: v12's H1 on band values; difference_interval: the paired band-minus-top interval."""
    if band["holds"]:
        return PERSISTS
    band_low, band_high = band["interval"]
    difference_low, difference_high = difference_interval
    if band_low <= 0 <= band_high and not difference_low <= 0 <= difference_high:
        return TOP50_SPECIFIC
    return UNCLEAR


def t2(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], sign: int, rng) -> dict:
    """groups: (log peak, peak quartile, band, top) per model, paired by neuron. Values are oriented by
    sign (higher = more mixed) before v12's H1 is applied, so its statistics are on that scale."""
    oriented = [(p, q, sign * np.asarray(b, float), sign * np.asarray(t, float)) for p, q, b, t in groups]
    band = h1([(p, q, b) for p, q, b, _ in oriented], rng)
    top = h1([(p, q, t) for p, q, _, t in oriented], rng)
    difference, interval = top_minus_rest([(q, b - t) for _, q, b, t in oriented], rng)
    return {"band": band, "top": top, "band_minus_top": {"difference": difference, "interval": list(interval)},
            "verdict": verdict(band, list(interval))}


def ols_coefficients(rows: list[dict], rng) -> dict:
    """S2: least squares of rows' "flag" field (here the oriented band coherence) on v12's standardized
    covariates with model fixed effects, and bootstrap intervals for the covariates' coefficients."""
    x, y = design(rows)
    x = np.column_stack([np.ones(len(y)), x])

    def fit(xs, ys):
        return np.linalg.lstsq(xs, ys, rcond=None)[0][1:1 + len(COVARIATES)]

    point = fit(x, y)
    draws = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        draws.append(fit(x[idx], y[idx]))
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    out = {c: {"coefficient": float(p), "interval": [float(lo), float(hi)]}
           for c, p, lo, hi in zip(COVARIATES, point, low, high)}
    out["holds"] = bool(high[0] < 0)
    return out
