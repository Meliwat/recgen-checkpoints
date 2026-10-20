"""Pre-registered v10 rules, written and committed before any v10 language-model reading.

v7 put three readers on the same units at the middle layer of the six sizes. Over all models,
Haiku flagged 50/50 pools of two random neurons 21 points (14 to 27) more often than real neurons,
about as well as the network reader (+22) and far better than MiniLM (+3). But Haiku was noisy
(retest flag kappa 0.37). v10 asks whether stronger one-shot language-model readers detect pooling
better. Only the reader changes. See README, "v10".

Readers: READERS, through the claude CLI with exactly v7's other flags (recgen/llm_reader.CLI with
the model swapped), on v7's units (recgen/units_v7.build_units), with v7's prompt and the same
blind snippets in the same seeded order (recgen/llm_reader.unit_prompt), and a second reading of
v7's 60 retest units (recgen/units_v7.retest_ids). v7 passed no effort flag and had no
CLAUDE_CODE_EFFORT_LEVEL set, so Haiku ran at the CLI's default effort for its model. v10 does the
same: no effort flag, and that variable is removed from the CLI's environment. The resolved model
ids come from the CLI's output and are stored with each answer.

Reading order: first readings of real neurons and 50/50 pools, then the retests, then the other
units (70/30, 90/10 and related), each tier in a seeded random order (reading_order). If the usage
budget stops a run early, the main comparison's units are done first. A reading that fails with a
CLI error is tried again on a later run; a reply with no valid label after v7's four attempts is
final and counts as missing.

Budget: no new reading starts once the account's weekly all-models meter reaches STOP_WEEKLY or the
5-hour window reaches STOP_FIVE_HOUR. Sonnet runs on every unit first, then Opus.

Pre-registered, per reader (Haiku's v7 answers analyzed the same way, for comparison):
1. v7's tests unchanged: power, specificity, agreement and scale (recgen/rubric_v7,
   recgen/analyze_v7), each reader in the "llm" slot.
2. The over-all-models 50/50 excess: the mean over the six models of (50/50 pooled flag rate minus
   real flag rate), with a 95% bootstrap interval that resamples units within each model and unit
   type, as recgen/analyze_v7.exploratory_stats does (excess_interval).
3. Retest: same label, same flag and flag kappa on the 60 retest units, with a bootstrap interval.
4. Agreement on real neurons: Cohen's kappa with Haiku and with the network reader, per model and
   over all 600 real neurons (with a bootstrap interval).
5. Label shares (A, B, C) on real neurons.

Main comparison, per stronger reader: does its over-all-models 50/50 excess beat Haiku's? The
difference (reader minus Haiku) is bootstrapped paired: each draw resamples the same units for
both readers, using only units both labeled (paired_difference). Verdict (compare): BETTER if the
95% interval's low end is above 0, WORSE if its high end is below 0, otherwise NO_DIFFERENCE. Each
direction is a one-sided test at 2.5%, so the chance of a false BETTER for either of the two readers
is about 5%. Planning estimate, from v7's interval for Haiku (about +-7 points): the paired
difference has a standard error of about 3.5 to 5 points, depending on how much two readers agree
unit by unit, so the comparison detects a true gain of about 10 to 14 points with 80% probability.
A smaller gain could be real and missed.

Secondary, same comparison with "two or more jobs" (B or C) as the flag (multiple_jobs), in case a
stronger reader finds a relation between two random neurons and answers B rather than C.

How a larger reader's lower retest noise is judged (decided before any reading):
- A unit's label can change between readings. Every unit is read once for the excess, so that
  variation doesn't bias the excess (each reading is an unbiased draw of the reader's flag rate on
  that unit); it only widens the interval. So the main comparison gives no credit for consistency:
  a reader with the same excess and less noise is more reliable, not more sensitive.
- Consistency is judged on its own: each reader's retest flag kappa beside Haiku's on the same 60
  units, with a paired bootstrap interval of the difference (paired_kappa_difference). With 60
  units this is a weak test, so it is descriptive.
- Noise in either reader pulls their agreement down, so a more consistent reader should agree more
  with every other reader even if it reads the same way. Each kappa is reported with the ceiling
  the two retests allow, sqrt(retest kappa a * retest kappa b), using 1 for the network reader,
  which is deterministic (kappa_ceiling). Higher agreement counts as more than lower noise only if
  kappa divided by its ceiling rises too. Kappa isn't a correlation, so this correction is rough
  and descriptive.
"""

import numpy as np

from recgen.rubric_v6 import cohen_kappa
from recgen.rubric_v7 import RELATED_JOBS, UNRELATED_JOBS

READERS = ("sonnet", "opus")
BASELINE = "haiku"
STOP_WEEKLY = 75  # percent of the weekly all-models limit
STOP_FIVE_HOUR = 85  # percent of the 5-hour limit
BOOTSTRAP = 2000
INTERVAL = 95
BETTER, WORSE, NO_DIFFERENCE = "better than Haiku", "worse than Haiku", "no clear difference"
MAIN_FIRST = ("real", "pooled 50/50")  # unit groups whose first readings come first


def reading_order(jobs: list[tuple[str, bool]], seed: int) -> list[int]:
    """Indices of jobs, given as (unit group, is retest), in reading order: first readings of the
    main comparison's groups, then retests, then the rest, each tier shuffled with the seed."""
    tiers = [[i for i, (g, again) in enumerate(jobs) if not again and g in MAIN_FIRST],
             [i for i, (_, again) in enumerate(jobs) if again],
             [i for i, (g, again) in enumerate(jobs) if not again and g not in MAIN_FIRST]]
    rng = np.random.default_rng([seed, 10])
    return [int(i) for tier in tiers for i in rng.permutation(np.array(tier, int))]


def multiple_jobs(label: str | None) -> bool | None:
    """The secondary flag: two or more jobs, related or not."""
    return None if label is None else label in (RELATED_JOBS, UNRELATED_JOBS)


def excess(pooled: list[np.ndarray], real: list[np.ndarray]) -> np.ndarray:
    """Each reader's mean over models of the pooled minus real flag rate. Each array holds one
    model's flags, [units, readers]."""
    return np.mean([p.mean(axis=0) - r.mean(axis=0) for p, r in zip(pooled, real)], axis=0)


def bootstrap_excess(pooled: list[np.ndarray], real: list[np.ndarray], rng, draws: int = BOOTSTRAP) -> np.ndarray:
    """[draws, readers] excesses; each draw resamples units within each model and unit type, the
    same units for every reader."""
    def resample(flags: np.ndarray) -> np.ndarray:
        return flags[rng.integers(0, len(flags), len(flags))]

    return np.array([excess([resample(p) for p in pooled], [resample(r) for r in real]) for _ in range(draws)])


def interval(draws: np.ndarray) -> list[float]:
    tail = (100 - INTERVAL) / 2
    return [float(x) for x in np.percentile(draws, [tail, 100 - tail])]


def excess_interval(pooled: list[np.ndarray], real: list[np.ndarray], rng) -> dict:
    """One reader's over-all-models excess and its interval, from per-model 1-D flag arrays."""
    pooled, real = [np.asarray(p, float)[:, None] for p in pooled], [np.asarray(r, float)[:, None] for r in real]
    return {"excess": float(excess(pooled, real)[0]), "interval": interval(bootstrap_excess(pooled, real, rng)[:, 0])}


def compare(low: float, high: float) -> str:
    if low > 0:
        return BETTER
    if high < 0:
        return WORSE
    return NO_DIFFERENCE


def paired_difference(pooled: list[np.ndarray], real: list[np.ndarray], rng) -> dict:
    """Reader minus Haiku in the over-all-models excess. Each array is one model's flags on units
    both readers labeled, [units, 2], Haiku first."""
    pooled, real = [np.asarray(p, float) for p in pooled], [np.asarray(r, float) for r in real]
    point = excess(pooled, real)
    draws = bootstrap_excess(pooled, real, rng)
    low, high = interval(draws[:, 1] - draws[:, 0])
    return {"haiku": float(point[0]), "reader": float(point[1]), "difference": float(point[1] - point[0]),
            "interval": [low, high], "verdict": compare(low, high)}


def kappa_interval(a: np.ndarray, b: np.ndarray, rng, draws: int = BOOTSTRAP) -> dict:
    """Cohen's kappa between two flag arrays on the same units, with a bootstrap interval."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    kappas = []
    for _ in range(draws):
        idx = rng.integers(0, len(a), len(a))
        kappas.append(cohen_kappa(a[idx], b[idx]))
    tail = (100 - INTERVAL) / 2
    return {"kappa": cohen_kappa(a, b), "interval": [float(x) for x in np.nanpercentile(kappas, [tail, 100 - tail])],
            "n": len(a)}


def paired_kappa_difference(haiku: tuple[np.ndarray, np.ndarray], reader: tuple[np.ndarray, np.ndarray], rng,
                            draws: int = BOOTSTRAP) -> dict:
    """Reader minus Haiku in retest flag kappa, (first, second) flags on the same units, resampled
    together."""
    (h1, h2), (r1, r2) = [tuple(np.asarray(x, bool) for x in pair) for pair in (haiku, reader)]
    diffs = []
    for _ in range(draws):
        idx = rng.integers(0, len(h1), len(h1))
        diffs.append(cohen_kappa(r1[idx], r2[idx]) - cohen_kappa(h1[idx], h2[idx]))
    tail = (100 - INTERVAL) / 2
    return {"difference": cohen_kappa(r1, r2) - cohen_kappa(h1, h2),
            "interval": [float(x) for x in np.nanpercentile(diffs, [tail, 100 - tail])]}


def kappa_ceiling(retest_a: float, retest_b: float = 1.0) -> float:
    """The agreement two readers could reach given their own retest kappas (1 for a deterministic
    reader); a rough attenuation bound."""
    return float(np.sqrt(max(retest_a, 0.0) * max(retest_b, 0.0)))
