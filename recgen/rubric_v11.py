"""Pre-registered v11 rules, written and committed before any v11 language-model reading.

v7 put three readers (MiniLM, the network's own representation, and Haiku) on the same units at
the middle layer only. There, MiniLM's falling share of mixed-looking neurons wasn't seen by the
two readers that can detect pooled neurons. The other depths are where v5 and v6 disagree most:
v5's MiniLM share rises with size at three quarters and drifts down at a quarter, and v6's network
reader loses power at three quarters in bigger models. v11 repeats v7's design, unchanged, at the
quarter and three-quarters depths of the same six sizes. See README, "v11".

Unchanged from v7 (recgen/rubric_v7.py): the units (N_REAL real, N_POOLED pooled at each of the
three mixes, N_RELATED related, per model), the three readers, the blind prompt, the CLI flags,
the labels, and the per-model tests. Only the depth changes. At each depth the same seeded draws
pick indices into that depth's sample of v5/v6's neurons, so v11's units at the middle depth are
exactly v7's (a test checks this), and v7's readings stand in for the middle depth below. Unit ids
add the depth, so each depth has its own cache entries and its own snippet order.

Pre-registered, at each of DEPTHS and for each reader:
1. Power, 2. specificity, 3. agreement (Cohen's kappa on real neurons, per model), 4. scale (v5's
   verdict rule on the real flag rate): v7's tests, unchanged.
5. Over-all-models excess: the mean over the six models of (50/50 pooled flag rate minus real flag
   rate), with a 95% bootstrap interval that resamples each model's units of each kind. In v7 this
   was an exploratory check added after review; here it is the main measure of power.

Questions across depths:
- How each reader's power varies with depth: for each reader and each pair in DEPTH_PAIRS (the
  middle depth from v7's readings), the difference in over-all-models excess, with a 95% interval
  from independent bootstrap draws at the two depths. The depths "differ" if that interval excludes
  zero. With COMPARISONS comparisons, a difference also "survives correction" if the Bonferroni
  interval (100 - 5 / COMPARISONS percent) excludes zero.
- Whether each depth's scale trend depends on the reader (reader_dependence): a reader is
  "sensitive" at a depth if its over-all-models excess interval lies above zero. A reader's trend
  verdict ("fits" or "against") is "corroborated" if another sensitive reader gives the same verdict.
  The depth's outcome is "no reader sees a trend" if every verdict is "no clear trend",
  "corroborated" if every trend verdict is corroborated, and otherwise "reader-dependent" (at the
  middle depth, v7's MiniLM "fits" with sensitive readers seeing no trend, this would be
  "reader-dependent").
- Descriptive, not tested: each reader's per-model excess against size, as a Spearman correlation
  with size rank, per depth (v6 found the network reader's excess falling with size at three
  quarters).

Retest: RETEST_PER_MODEL units per model and depth (the same seeded positions as v7's) are read a
second time; per depth, the share with the same label and the same flag, and the flag kappa with a
95% bootstrap interval.

Checks: on real neurons, v11's MiniLM and network flags must equal v5's and v6's flags for the
same neurons, since those readers are unchanged. The language model must resolve to the model v7
used (V7_RESOLVED_MODEL); any other model is reported. A unit with no valid label after the
reader's retries drops out of Haiku's rates and is counted (as in v7). If plan usage limits stop the
run early, the analysis uses only the cached readings and lists how many units are missing.
"""

import numpy as np
from scipy.stats import spearmanr

from recgen.config import SEED
from recgen.rubric_v5 import AGAINST, FITS, INTERVAL

DEPTHS = ("quarter", "three quarters")
ALL_DEPTHS = ("quarter", "middle", "three quarters")  # the middle depth is v7's
DEPTH_PAIRS = (("quarter", "middle"), ("three quarters", "middle"), ("quarter", "three quarters"))
N_READERS = 3
COMPARISONS = len(DEPTH_PAIRS) * N_READERS
EXCESS_BOOTSTRAP = 2000
RETEST_PER_MODEL = 10
V7_RESOLVED_MODEL = "claude-haiku-5-5"

NO_READER_TREND = "no reader sees a trend"
CORROBORATED = "corroborated"
READER_DEPENDENT = "reader-dependent"


def depth_key(depth: str) -> str:
    return depth.replace(" ", "_")


def rng_for(depth: str, purpose: int):
    return np.random.default_rng([SEED, 11, ALL_DEPTHS.index(depth), purpose])


def _interval(draws: np.ndarray, percent: float = INTERVAL) -> list[float]:
    tail = (100 - percent) / 2
    return [float(x) for x in np.percentile(draws, [tail, 100 - tail])]


def excess_over_models(per_model: list[tuple[np.ndarray, np.ndarray]], rng, draws: int = EXCESS_BOOTSTRAP) -> dict:
    """per_model: (50/50 pooled flags, real flags) for each model. The mean over models of the pooled
    minus the real flag rate, its interval, and the bootstrap draws behind it."""
    point = float(np.mean([np.mean(p) - np.mean(r) for p, r in per_model]))
    samples = np.zeros(draws)
    for pooled, real in per_model:
        pooled, real = np.asarray(pooled, float), np.asarray(real, float)
        samples += rng.choice(pooled, (draws, len(pooled))).mean(axis=1) - rng.choice(real, (draws, len(real))).mean(axis=1)
    samples /= len(per_model)
    return {"excess": point, "interval": _interval(samples), "draws": samples}


def compare_depths(a: dict, b: dict) -> dict:
    """Depth a's over-all-models excess minus depth b's, from excess_over_models results drawn
    independently."""
    draws = a["draws"] - b["draws"]
    interval, corrected = _interval(draws), _interval(draws, 100 - (100 - INTERVAL) / COMPARISONS)
    return {"difference": a["excess"] - b["excess"], "interval": interval, "differ": not interval[0] <= 0 <= interval[1],
            "corrected_interval": corrected, "survives_correction": not corrected[0] <= 0 <= corrected[1]}


def reader_dependence(verdicts: dict[str, str], excess_intervals: dict[str, list[float]]) -> dict:
    """verdicts: each reader's scale verdict at one depth; excess_intervals: each reader's
    over-all-models excess interval there."""
    sensitive = [r for r, (low, _) in excess_intervals.items() if low > 0]
    trends = {r: v for r, v in verdicts.items() if v in (FITS, AGAINST)}
    corroborated = {r: [o for o in sensitive if o != r and verdicts[o] == v] for r, v in trends.items()}
    if not trends:
        outcome = NO_READER_TREND
    elif all(corroborated.values()):
        outcome = CORROBORATED
    else:
        outcome = READER_DEPENDENT
    return {"outcome": outcome, "sensitive": sensitive, "corroborated_by": corroborated,
            "verdicts": dict(verdicts)}


def excess_against_size(per_model_excess: list[float]) -> float:
    """Descriptive: Spearman correlation of a reader's per-model excess with size rank, smallest first."""
    return float(spearmanr(np.arange(len(per_model_excess)), per_model_excess).statistic)

