"""Pre-registered v20 rules, written and committed before any v20 recording or reading.

Question. v17 read 50 examples per neuron from a band at a fixed fraction of each neuron's own peak
([0.40, 0.60)) and found that strongly firing neurons' band examples are still more coherent (v6's
top-pair percentile) and less often called unrelated jobs by Haiku than weakly firing neurons'. Its
weakest point: the same fraction of peak is not the same selectivity. A weak neuron's band holds 44 to
53% of all windows, a strong neuron's 15 to 25%, so weak neurons' band examples are close to random
text, and "broader tuning" fits the persisting effect as well as "more mixing". v20 matches selectivity
by rank instead: every neuron's examples come from the same ranks of its own 4,096 window maxima, so
every sample is equally selective (the same number of windows rank above it). The top 50 is already a
rank-matched sample (ranks 1 to 50); v20 asks whether the effect holds at matched rank below the top.

Bands. A neuron's windows are ranked by v5's window maximum (data["max"], as float32), largest first,
ties broken by window index (a stable sort). RANKS gives each band's first and last rank, inclusive,
1 being the largest:
- "band" (primary): ranks 201 to 500.
- "band2" (secondary): ranks 1,001 to 2,000, decided here, before any recording.
v6's top-50 windows are left out of every band, so band and top-50 samples never share a window. Per
neuron and band, BAND_SIZE windows are drawn uniformly without replacement with rank_rng(model, neuron
id, band), one example per window at the window's strongest position (v5's data["pos"]), as for the top
50. A neuron is excluded from a band, in both halves of every pair, if its window maximum at the band's
last rank is not above 0 (it doesn't fire in every band window) or if fewer than BAND_SIZE band windows
remain. Each sampled example's fraction of the neuron's peak (its window maximum over the neuron's
largest) is stored, so the overlap with v17's band can be described.

Before this file was committed, these were computed from v5's window maxima only (no residual
recorded, no detector run, no reading made): no sampled neuron is excluded from "band"; 10 of 11,801
are excluded from "band2" (2, 3, 0, 2, 3, 0 by model); none of v12's 600 fresh neurons is excluded from
either; no top-50 window falls in "band". The median example of "band" sits at 0.62 to 0.65 of the
peak in the weakest peak quartile and 0.49 to 0.59 in the strongest (range over models), and 0 to 25%
of the weakest quartile's "band" windows fall in v17's band, against 51 to 100% of the strongest's.
"band2" sits at 0.46 to 0.50 (weakest) and 0.30 to 0.38 (strongest), with 100% and 0 to 38% in v17's
band. So at matched rank, weak neurons' examples are at a higher fraction of their own peak than strong
neurons', the opposite of v17's band.

Disk. data/v20 is capped at 2 GB. The residuals at "band" positions are stored as v17 stored its band
(about 1.6 GB, float16). "band2" positions are recorded in the same forward pass, read by the network
reader in memory right after recording, and only its per-neuron readings are stored.

Readers, unchanged from v17.
- network: v6's detector at its pre-registered settings (recgen.analyze_v17.read_examples): the residual
  stream entering the layer, LayerNorm-normalized (recgen.sweep_v6.normalized_residuals), centered
  with v6's random-position mean, the neuron's folded input direction projected out, against v6's null
  for that neuron. Per neuron and sample: flag (unrelated mixed), n_facets, coherence (v6's
  top_pair_percentile). Top-50 values are recomputed from v6's recordings and must equal v6's
  published values, and activations rebuilt from the recorded residuals are compared with v5's window
  maxima at every band example of both bands (the rebuild error). Neurons: all of v6's sampled
  neurons at DEPTH. As in v17, the detector also reads BAND_SIZE of v6's random positions per neuron
  (recgen.rubric_v17.reference_rng) as a reference, not a test.
- Haiku: v12's 600 fresh neurons (recgen.read_v12.fresh_units), each shown as its "band" examples and,
  separately, its "band2" examples, exactly as v7 shows a unit (recgen.llm_reader: same prompt, CLI,
  marked token, no activations, order seeded by the unit id), so each has a top-50 reading (v12's) and
  two rank-band readings. RETEST of the "band" units, drawn with rng_for, are read a second time. Drift
  check, as in v17: the same RETEST neurons' top-50 prompts, byte-identical to v12's, are read again.
  At most WORKERS calls at once. Reading stops once the CLI-reported cost of v20's readings reaches
  SPEND_LIMIT_USD (recgen.read_v17.SpendGuard).

Tests, per band, reader and measure (network flag, network coherence, Haiku flag), v17's unchanged
(recgen.rubric_v17), with MIXED_SIGN orienting each measure so higher means more mixed:
- T1, within neuron: band minus top 50, paired by neuron, each model's mean averaged over the six
  models, with a bootstrap interval (v17's t1).
- T2, the magnitude effect in the band: v12's H1 on band values (strongest peak quartile minus the
  other three, interval below 0 on the mixed side, Spearman of log peak negative on the mixed side in at
  least 5 of 6 models), the same statistic on the same neurons' top 50, and the paired band-minus-top
  difference (v17's t2). Peak quartiles are among all sampled neurons of the layer, and for Haiku v12's
  quartile of each fresh neuron.
- Verdict: v17's rule (verdict): PERSISTS if T2 holds in the band; TOP50_SPECIFIC if the band
  statistic's interval includes 0 and the paired difference's excludes it; otherwise UNCLEAR.

How each outcome is read, on "band" (primary):
- PERSISTS: at matched selectivity, strong neurons' examples still look more coherent or less mixed.
  The magnitude effect is then a property of the neuron beyond selectivity, and v17's persistence was
  not produced by the difference in band width.
- TOP50_SPECIFIC: the effect is present in the top 50 and absent at matched rank below it. v17's
  persistence below the top was then explained by selectivity (its weak neurons' band was the less
  selective one), and the effect is confined to each neuron's very top.
- UNCLEAR: neither.
Coherence and the Haiku flag are co-primary, as in v17. The answer to v20's question (answer) is YES
if both their "band" verdicts are PERSISTS, NO if both are TOP50_SPECIFIC, otherwise MIXED. The network
flag is tested and reported the same way, with v17's caveat: it needs two facets of at least 5
examples, so it can fall where examples are more diffuse. "band2" gets the same tests and verdicts as
a secondary result: whether the answer holds further down each neuron's range.

Comparison with v17's band (descriptive, no verdict), on the neurons included in both:
- by peak quartile, each measure in the top 50, v17's band, "band" and "band2";
- v12's H1 statistic (oriented) in v17's band and in "band", and the paired difference "band" minus
  v17's band, with a bootstrap interval (against_v17);
- by peak quartile, the median fraction of peak of "band" and "band2" examples, the share of them in
  v17's band, and the median rank of v17's band examples (range over models).

Secondary, pre-registered but not part of the verdict, as v17's S1 and S2 on "band": v12's H2 on band
flags (both readers) and v17's least-squares version on oriented band coherence, with v12's covariates
(log peak, peakedness, density) and the band examples' token purity.

Spend guard: SPEND_LIMIT_USD, CLI-reported, over all v20 readings.
"""

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import MODELS
from recgen.rubric_v17 import BAND_SIZE, DEPTH, MIXED_SIGN, PERSISTS, TOP50_SPECIFIC, UNCLEAR, t1, t2, verdict  # noqa: F401

RANKS = {"band": (201, 500), "band2": (1001, 2000)}  # first and last rank, inclusive; 1 is the largest
PRIMARY_BAND = "band"
RANK_SEED = 20
RETEST = 30  # 5 per model
SPEND_LIMIT_USD = 10.0
WORKERS = 5  # at most 6 CLI calls at once

READINGS = {PERSISTS: "a property of the neuron beyond selectivity",
            TOP50_SPECIFIC: "v17's persistence explained by selectivity",
            UNCLEAR: "unclear"}
YES, NO, MIXED = "yes", "no", "mixed"


def rank_rng(model: str, neuron_id: int, band: str):
    """The generator that draws one neuron's windows for one band."""
    return np.random.default_rng([SEED, RANK_SEED, 0, list(RANKS).index(band), MODELS.index(model), neuron_id])


def rng_for(model: str, purpose: int):
    return np.random.default_rng([SEED, RANK_SEED, 2, MODELS.index(model), purpose])


def answer(coherence_verdict: str, haiku_verdict: str) -> str:
    """v20's answer from the two co-primary verdicts on the primary band."""
    if coherence_verdict == haiku_verdict == PERSISTS:
        return YES
    if coherence_verdict == haiku_verdict == TOP50_SPECIFIC:
        return NO
    return MIXED


def against_v17(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], sign: int, rng) -> dict:
    """Descriptive: groups are (log peak, peak quartile, rank band, v17 band) per model, paired by neuron.
    v12's H1 in each band (oriented by sign) and the paired difference rank band minus v17's band."""
    out = t2(groups, sign, rng)
    return {"rank_band": out["band"], "v17_band": out["top"], "rank_minus_v17": out["band_minus_top"]}
