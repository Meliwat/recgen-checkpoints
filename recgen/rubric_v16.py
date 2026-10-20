"""Pre-registered v16 rules, written and committed before any v16 reading.

The essay under test (Nick Cammarata, "Neural Networks Are Already Clean", Goodfire, 2026-10-15,
https://already-clean.goodfire.com/, "The Short Version" and its footnote 1) reports a preliminary
audit of Qwen3-32B neuron poles: roughly 90% have "one main job", often with "small related
offshoots", about 7% have "two related jobs", and about 2 to 4% are "not yet understood". Its
footnote says these numbers depend on how far into weak activations one looks, the depth of the
analysis, what counts as "one thing", and whether activation percentiles are used. Its scheme has no
category for two unrelated jobs. v7 gave one-shot Haiku another scheme (A one job, B related jobs,
C unrelated jobs): Haiku called 42 to 57% of real neurons one job, and flagged 50/50 pools of two
random neurons about 21 points more often than real neurons. v16 asks:

1. Given to the same one-shot readers on v7's units, does the essay's scheme push the share of real
   neurons called one main job toward 90%?
2. Does it then still separate known two-neuron pools from real neurons?
3. Does adding an explicit "unrelated jobs" option change that?

This tests the rubric's power to tell known mixtures from real neurons, with Pythia's top-50
snippets and one-shot readers. It does not test the essay's number, which comes from agentic
investigation of Qwen3-32B neuron poles. See README, "v16".

Units: exactly v7's (recgen/units_v7.build_units): the middle layer of all six sizes, per size 100
real neurons, 50 pools at each mix (50/50, 70/30, 90/10) and 50 related pools (1,800 units), and
v7's 60 retest units (recgen/units_v7.retest_ids), read a second time. Each unit is shown exactly as
v7 shows it (recgen/llm_reader.unit_prompt): the same 50 snippets in the same seeded order, with the
firing token marked and no activation values.

Arms (ARMS), two rubrics by two readers:
- Rubrics: v7's prompt with only its decision paragraph (V7_DECISION) replaced by DECISIONS[rubric].
  The JSON reply, a label and a list of patterns, is unchanged.
  - E3, the essay's scheme: A ONE MAIN JOB, B TWO RELATED JOBS, C NOT YET UNDERSTOOD.
  - E4: E3 plus D TWO OR MORE UNRELATED JOBS.
  Letters mean different things in v7's rubric, where C is "unrelated jobs"; LABEL_NAMES has both.
- Readers: haiku (v7's reader, the primary one) and sonnet (v10's), through the claude CLI with v7's
  system prompt and flags (recgen/llm_reader.CLI with only the model swapped). The resolved model id
  the CLI reports is stored with every answer. Two settings differ from v7, the same for every unit:
  - The CLI's prompt caching is off (DISABLE_PROMPT_CACHING=1). Each prompt is read once, and a
    check before any reading, on a filler prompt and no unit (data/v16/precheck_caching.jsonl),
    showed the cache write doubles the CLI-reported cost. The model sees the same input either way.
  - Environment variables that name an effort level are removed from the CLI's environment, as v10
    does, so the CLI runs at its default effort, as in v7. (The same check found that
    CLAUDE_EFFORT=xhigh, set in the session that runs v16, didn't change Haiku's thinking length.)
  The CLI runs in the repo folder, as in v7 and v10.

Flags:
- Primary: "not one main job" (not_one_main_job), any label other than A, in both rubrics.
- E4 only, secondary: "unrelated jobs" (unrelated_jobs), label D alone.
- Ordered labels for the AUC (LABEL_RANK): A < B < C in E3, and A < B < C < D in E4, where D ranks
  above "not yet understood" because it is the answer that names a mixture. v7's rubric: A < B < C.

Reading order (reading_order): Haiku's readings, then Sonnet's. Per reader: the first readings of
real neurons and 50/50 pools, then the 60 retests, then the other units (70/30, 90/10 and related),
each tier in a seeded random order, with each unit's E3 and E4 readings next to each other.

Spend guard: SPEND_CAP_USD over every v16 CLI call (the caching check and the pilot included), by the
CLI's reported cost. A reading starts only if the spend so far, plus RESERVE_USD for it and for each
reading in flight, stays within the cap. If the guard trips, no new reading starts and the analysis
reports what finished. A reading that ends in a CLI error is tried again on a later run; one with no
valid label after v7's four attempts is final and counts as missing. Missing units are left out of
that arm's rates, and pairwise out of paired comparisons. Expected cost (from v7, v10 and the
caching check): about $1 per Haiku arm and $18 per Sonnet arm.

Pilot (PILOT_*): before any v16 unit is read, three units outside v7's sample (pythia-160m at the
quarter depth: the first two real neurons and the first 50/50 pool, built as v7 builds its units)
are read once in every arm, into data/v16/pilot/, to check that the replies parse. The prompt is
amended only if something is broken, with a dated and committed reason, before any v16 unit.

Baselines, v7's rubric on the same units by the same reader:
- Haiku: v7's first readings (results/v7_llm_answers.json).
- Sonnet: v10's first readings (data/v10/sonnet/llm/), used only if every one of the 1,800 units
  has a cached reading of v7's exact prompt (its SHA-256). Otherwise Sonnet has no baseline and its
  tests against v7's rubric are reported as not run.

Statistics: "over sizes" is the mean over the six sizes of each size's value. Intervals are 95%
percentile bootstraps (BOOTSTRAP draws, bootstrap) that resample units within each size and unit
type; comparisons between rubrics or arms are paired (each draw resamples the same units for both)
and use the units both labeled. Retest statistics resample the 60 retest units.

Pre-registered tests, per arm:
1. One-main-job share (label A) of real neurons, over sizes and per size, with intervals. Compared
   with the essay's ESSAY_ONE_MAIN_JOB (near_essay: the interval includes 0.90 or lies above it) and
   with the same reader's A share under v7's rubric on the same units (rise_real, paired).
2. Pool detection: the 50/50 pools' primary flag rate minus the real neurons', over sizes, with an
   interval; and the label AUC (LABEL_RANK) for ranking 50/50 pools above real neurons, over sizes,
   with an interval. For E4, detection with the "unrelated jobs" flag too.
3. One-main-job share of 50/50 pools, and of 70/30 and 90/10 pools (dose-response), with intervals.
   Related pools are reported beside them, descriptively.
4. Retest on the 60 retest units: same label, same primary flag, and the flag's Cohen's kappa with
   an interval.

Verdict, per reader and rubric, against v7's rubric (verdict). rise_real and rise_pool are the
paired rises (this rubric minus v7's) in the one-main-job share of real neurons and of 50/50 pools,
over sizes, and diff = rise_pool - rise_real. Because "not one main job" is 1 minus "one main job",
diff is minus the change in detection with that flag.
- INFLATES, "inflates cleanliness without discrimination": rise_real's interval lies above 0 (its
  share exceeds the v7-rubric A share; judged by the interval so that noise can't trigger it) and
  diff's interval includes 0 or lies above 0 (pools rise at least as much, within noise).
- SELECTIVE, "raises cleanliness, less for known pools": rise_real's interval lies above 0 and
  diff's interval lies below 0, so detection with the not-one-main-job flag improves.
- NO_RISE: rise_real's interval includes 0.
- LOWERS: rise_real's interval lies below 0.
E3 is the primary rubric for the verdict and Haiku the primary reader; E4 and Sonnet are secondary.

E4 against E3 (question 3), per reader, paired: the differences (E4 minus E3) in the real neurons'
one-main-job share, the 50/50 pools' one-main-job share, and detection with the primary flag, each
"changes" if its interval excludes 0 (changes). Also, E4's detection with "unrelated jobs" against
the same reader's v7-rubric detection with C (unrelated jobs), paired.

Anything else in the analysis is labeled exploratory.

Clarifications added on 2026-10-20, after the run and an independent review (no rule changed):
- "agentic investigation" above: the essay doesn't say how its preliminary audits of Qwen3-32B neuron
  poles were run; read it as "a preliminary audit whose method the essay doesn't detail".
- Footnote 1's factors above: it says "depth" without saying of what (not "the depth of the
  analysis"), and it also names the density of sub-circuits.
- "as v10 does" above: v10 removed only CLAUDE_CODE_EFFORT_LEVEL and kept the CLI's prompt caching;
  v16 removes every variable whose name contains EFFORT, CLAUDE_EFFORT among them. The check that
  CLAUDE_EFFORT=xhigh didn't change thinking covered Haiku only, two filler readings per condition,
  from a script that wasn't committed.
"""

import json
import re

import numpy as np

from recgen.rubric_v7 import INSTRUCTIONS as V7_INSTRUCTIONS

READERS = ("haiku", "sonnet")
RUBRICS = ("E3", "E4")
ARMS = {f"{reader}-{rubric}": (reader, rubric) for reader in READERS for rubric in RUBRICS}
PRIMARY_READER, PRIMARY_RUBRIC = "haiku", "E3"
V7_RUBRIC = "v7"

ONE_MAIN_JOB, TWO_RELATED_JOBS, NOT_YET_UNDERSTOOD, UNRELATED_JOBS = "A", "B", "C", "D"
LABELS = {"E3": ("A", "B", "C"), "E4": ("A", "B", "C", "D"), V7_RUBRIC: ("A", "B", "C")}
LABEL_NAMES = {
    "E3": {"A": "one main job", "B": "two related jobs", "C": "not yet understood"},
    "E4": {"A": "one main job", "B": "two related jobs", "C": "not yet understood", "D": "unrelated jobs"},
    V7_RUBRIC: {"A": "one job", "B": "related jobs", "C": "unrelated jobs"},
}
LABEL_RANK = {rubric: {label: rank for rank, label in enumerate(labels)} for rubric, labels in LABELS.items()}
V7_UNRELATED = "C"  # v7's rubric's flag

V7_DECISION = """Decide which describes what the unit responds to:
A) ONE JOB: all or nearly all snippets share one pattern. The pattern may be broad or abstract, such as a grammatical role, a spelling fragment, or a kind of context.
B) RELATED JOBS: two or more patterns that are clearly related to each other.
C) UNRELATED JOBS: two or more patterns with no clear relation to each other."""
E3_DECISION = """Decide which describes what the unit responds to:
A) ONE MAIN JOB: one main job, often with small related offshoots. The main job may be broad or abstract, such as a grammatical role, a spelling fragment, or a kind of context.
B) TWO RELATED JOBS: two jobs that are clearly related to each other.
C) NOT YET UNDERSTOOD: no description of what the unit responds to that you are confident in."""
DECISIONS = {
    "E3": E3_DECISION,
    "E4": E3_DECISION + "\nD) TWO OR MORE UNRELATED JOBS: two or more jobs with no clear relation to each other.",
}

ESSAY_ONE_MAIN_JOB = 0.90
SPEND_CAP_USD = 70.0
RESERVE_USD = 0.05  # per reading in flight; about twice a Sonnet reading's cost with caching on
BOOTSTRAP = 2000
INTERVAL = 95
SEED_OFFSET = 16
MAIN_FIRST = ("real", "pooled 50/50")  # unit groups whose first readings come first
PILOT_MODEL, PILOT_DEPTH = "160m", "quarter"
PILOT_UNITS = {"real": 2, "pooled 50/50": 1}  # the first units of each group, in v7's build order

INFLATES = "inflates cleanliness without discrimination"
SELECTIVE = "raises cleanliness, less for known pools"
NO_RISE = "no clear rise in cleanliness"
LOWERS = "lowers cleanliness"


def instructions(rubric: str) -> str:
    """v7's instructions with only the decision paragraph replaced."""
    assert V7_DECISION in V7_INSTRUCTIONS
    return V7_INSTRUCTIONS.replace(V7_DECISION, DECISIONS[rubric])


def prompt(rubric: str, snippets: list[str]) -> str:
    """The rubric's prompt, numbering the snippets as v7 does."""
    return instructions(rubric).format(n=len(snippets), snippets="\n".join(f"{i + 1}. {s}" for i, s in enumerate(snippets)))


def parse_label(text: str, rubric: str) -> str | None:
    """The label from a reply, or None if there isn't one valid in this rubric (v7's parsing)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            label = json.loads(match.group(0)).get("label")
        except (json.JSONDecodeError, AttributeError):
            label = None
        if label in LABELS[rubric]:
            return label
    return None


def not_one_main_job(label: str | None) -> bool | None:
    """The primary flag."""
    return None if label is None else label != ONE_MAIN_JOB


def unrelated_jobs(label: str | None) -> bool | None:
    """E4's secondary flag."""
    return None if label is None else label == UNRELATED_JOBS


def reading_order(jobs: list[tuple[str, bool]], seed: int) -> list[int]:
    """Indices of one reader's jobs, given as (unit group, is retest), in reading order: first
    readings of the main groups, then retests, then the rest, each tier shuffled with the seed."""
    tiers = [[i for i, (g, again) in enumerate(jobs) if not again and g in MAIN_FIRST],
             [i for i, (_, again) in enumerate(jobs) if again],
             [i for i, (g, again) in enumerate(jobs) if not again and g not in MAIN_FIRST]]
    rng = np.random.default_rng([seed, SEED_OFFSET])
    return [int(i) for tier in tiers for i in rng.permutation(np.array(tier, int))]


def interval(draws) -> list[float]:
    tail = (100 - INTERVAL) / 2
    return [float(x) for x in np.nanpercentile(np.asarray(draws, float), [tail, 100 - tail])]


def bootstrap(strata: dict, statistic, rng, draws: int = BOOTSTRAP) -> dict:
    """statistic(strata) and its percentile interval. strata maps a key (a size and unit group) to
    an array [units, columns]; each draw resamples the units of every stratum, the same units for
    every column, so comparisons between columns are paired."""
    point = statistic(strata)
    values = [statistic({k: v[rng.integers(0, len(v), len(v))] for k, v in strata.items()}) for _ in range(draws)]
    return {"value": float(point), "interval": interval(values)}


def auc(pooled: np.ndarray, real: np.ndarray) -> float:
    """The chance a random pool scores above a random real neuron, ties counting half."""
    diff = np.asarray(pooled, float)[:, None] - np.asarray(real, float)[None, :]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))


def verdict(rise_real: list[float], diff: list[float]) -> str:
    """The pre-registered verdict from the intervals of rise_real and of rise_pool - rise_real."""
    if rise_real[0] > 0:
        return INFLATES if diff[1] >= 0 else SELECTIVE
    return LOWERS if rise_real[1] < 0 else NO_RISE


def near_essay(share_interval: list[float]) -> bool:
    """Whether a one-main-job share's interval includes the essay's 90% or lies above it."""
    return share_interval[1] >= ESSAY_ONE_MAIN_JOB


def changes(difference_interval: list[float]) -> bool:
    return difference_interval[0] > 0 or difference_interval[1] < 0
