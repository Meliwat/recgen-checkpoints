"""Pre-registered v24 rules, written and committed before any v24 reading.

v7's one-shot readers label a unit's 50 snippets (firing token marked, no activation values). Under
v7's rubric (A one job, B related jobs, C unrelated jobs), Haiku calls about 45% of real Pythia
neurons C and about 66% of known 50/50 pools of two random neurons. v20's post hoc check found it
calls 76% (67 to 84%) of units made of 50 random corpus positions C, more than pools, so C partly
tracks how diffuse a set of snippets is. Under the essay's categories (E3: A one main job, often with
small related offshoots; B two related jobs; C not yet understood), real neurons get 33 to 58% one
main job depending on the reader (v16, v21), known pools 11 to 32%, and the essay reports about 90%
for Qwen3-32B neuron poles. v24 asks:

    What do the one-shot readers say about pure random text, under each rubric?

It is a negative control. If a reader calls a sizable share of random-text units one job (or one
main job), its one-job share of real neurons, and any share like the essay's ~90%, has to be read
against that floor. And where real neurons sit between random text and known pools on each share
tells what the labels measure. See README, "v24".

Units (300, the middle layer):
- random: v20's 100 random-text units (recgen/read_posthoc_v20.random_units on branch v20),
  rebuilt exactly here (random_units): per size (17, 17, 17, 17, 16, 16 over rubric_v5.MODELS),
  RANDOM_UNIT_SIZE of v6's 1,024 random positions (one per window, the same positions for every
  size), drawn without replacement with v20's seed (V20_RANDOM_SEED), the marked token at the random
  position. Their v7-rubric prompts must hash to v20's (results/v20_posthoc_llm_answers.json on
  branch v20); the runner checks every one before reading and stops otherwise.
- real: N_PER_GROUP of v7's real neurons (recgen/units_v7.build_units), the same per-size counts,
  a seeded draw within each size (anchor_ids).
- pooled 50/50: N_PER_GROUP of v7's 50/50 pools of two random neurons, drawn the same way.
Every unit is shown exactly as v7 shows it (recgen/llm_reader.unit_prompt): the same 50 snippets
in the same seeded order (seeded by the unit id), the firing token marked, no activation values.

Arms (ARMS): three readers by two rubrics, six series.
- Readers: claude-haiku-5-5, claude-sonnet-5-5 and claude-opus-5-5, by full id. The resolved model
  id the CLI reports is stored with every reading.
- Rubrics: v7's prompt exactly (rubric_v7.prompt) and E3 exactly (rubric_v16.prompt("E3", ...)),
  through reader_v21.prompt_for, so every anchor prompt hashes to v7's and v16's for the same unit.
- CLI: v21 arm A's (and v16's) settings, through recgen/llm_reader_v16.call_cli: v7's system
  prompt and flags with only the model changed, the CLI's prompt caching off, environment variables
  naming an effort level removed (the CLI's default effort), the CLI run in the repo folder.
  Thinking tokens are recorded with every reading. At most WORKERS calls at once.
- Labels and parsing: rubric_v16.parse_label with the series' rubric (A, B or C in both).

Order (reading_order): every first reading (300 units x 6 series = 1,800) in one seeded random
order, all six series interleaved in one run, so the comparisons between readers and rubrics are
concurrent. Then the retests: RETEST_PER_GROUP units of each group (retest_ids, the same 30 units
for every series), read a second time in every series, in a seeded random order. If the spend guard
trips during the first readings, what was read is a random subset of every series.

Spend guard: SPEND_CAP_USD of CLI-reported cost over every v24 call (pilot, readings, retries),
with v16's guard (a reading starts only if the spend so far plus rubric_v16.RESERVE_USD for it and
for each reading in flight stays within the cap). Expected from v16's and v21's costs: about $0.5
for Haiku, $7 for Sonnet and $13 for Opus over the first readings, $2 for the retests. A reading
that ends in a CLI error is read again on a later run; one with no valid label after v7's four
attempts is final and missing. Missing units are left out of that series, and pairwise out of
paired comparisons.

Pilot (PILOT_UNITS): before any v24 unit is read, two units outside the sample, one random-text
unit built as v20 builds them but from pythia-160m's quarter-depth random positions with another
seed (PILOT_RANDOM_PURPOSE), and v16's first pilot real neuron (pythia-160m, quarter depth), are
read once in every series (12 readings), to check the CLI, the model ids and the parsing and to
estimate cost. The rules change only if something is broken, by a committed reason, before the first
v24 unit is read.

Statistics (v16's): "over sizes" is the mean over the six sizes of each size's value. Intervals are
95% percentile bootstraps (rubric_v16.bootstrap, BOOTSTRAP draws) that resample units within each
size and group; statistics that combine groups or series use the same draws, so comparisons between
series on the same units are paired. The label rank for AUCs is A < B < C in both rubrics
(rubric_v16.LABEL_RANK): under v7's rubric one job < related jobs < unrelated jobs, under E3 one
main job < two related jobs < not yet understood. AUC(X over Y) is the chance a unit of group X gets
a higher-ranked label than a unit of group Y of the same size, ties counting half, averaged over sizes.

Pre-registered outcomes, per series (reader and rubric):
P1. Floor: the share of random-text units called A (one job, or one main job), with an interval,
    and its reading (floor_verdict): SIZABLE_FLOOR if the interval's low end is at least
    FLOOR_SIZABLE; NO_SIZABLE_FLOOR if its high end is below FLOOR_SIZABLE; otherwise UNCLEAR_FLOOR.
    Reading: a sizable floor means the reader calls structureless text one (main) job often enough
    that its one-job share of real neurons can't be read as a share of neurons with one job; it has
    to be read as an excess over the floor (P4). No sizable floor means the label isn't handed to
    diffuse sets, so that kind of inflation isn't what lifts the real neurons' share.
P2. The C share of random-text units (unrelated jobs under v7's rubric, not yet understood under
    E3), with an interval, and the B share. Reading: under E3, "not yet understood" is the correct
    answer for random text, so its C share is how often the reader uses that category when nothing
    is there to understand; under v7's rubric, C on random text at or above the pools' C share
    means C rises with how diffuse a set is, not only with how many jobs it mixes (v20's reading).
P3. Anchor, read in the same run: the A, B and C shares of the 100 real neurons and the 100 50/50
    pools, with intervals.
P4. Gaps from the floor: real minus random and pools minus random, on the A share and on the C
    share, with intervals.
P5. Position of real neurons (position): p = (real - random) / (pools - random), on the A share and
    on the C share. It is defined only if the interval of pools - random excludes 0; its interval
    is the percentile interval of p over the bootstrap draws. Readings (position_reading):
    - REAL_LIKE_POOLS, p's interval includes 1: on this share real neurons look like known
      two-neuron pools, and both differ from random text, so the share separates structured sets
      from structureless text but not one job from two: it tracks how diffuse a set is.
    - REAL_BEYOND_POOLS, p's interval lies above 1: real neurons sit farther from random text than
      pools do, so the share separates one job from two, at least partly. 1/p is the part of the
      way from random text to real neurons that pools cover; the smaller it is, the more pools look
      like random text, and the more the share responds to mixing rather than to diffuseness.
    - REAL_BETWEEN, p's interval lies within [0, 1): real neurons look more like random text than
      known pools do on this share.
    - REAL_BELOW_RANDOM, p's interval lies below 0: real neurons look further from one job than
      random text does.
    - UNDEFINED_POSITION, the interval of pools - random includes 0: pools can't be told from
      random text on this share; it treats a two-neuron mixture like structureless text, so where
      real neurons differ from random text, the share responds to mixing.
    - UNCLEAR_POSITION: any other interval.
P6. Label AUCs (A < B < C), with intervals: random over real, random over pools, and pools over
    real (the anchor, v7's and v16's detection measure), each "separates" if its interval lies above
    0.5 (rubric_v21.separates). Reading: if random over real is well above pools over real, the labels
    separate structureless text from real neurons better than known mixtures from real neurons; if
    random over pools is near 0.5, the labels treat known pools like random text.

Secondary:
S1. Per reader, paired on the same units, E3 minus v7's rubric in the A share of random text, real
    neurons and pools: does E3's room for offshoots raise the floor too (v21 found it raises real
    neurons and pools alike for Opus)?
S2. Floor-corrected one-job share of real neurons, (real - random) / (1 - random) on the A share:
    the part of the room above the floor that real neurons take, beside the essay's ~90%.
S3. Retest on the 30 retest units per series: same label, same "not one main job" flag, the flag's
    Cohen's kappa with an interval (resampling units).
S4. Haiku under v7's rubric on the random-text units against v20's readings of the same prompts
    (another round, v7's CLI settings): same label share and the paired difference in A and C shares.
S5. Anchor units against earlier rounds of the same prompts: v7's Haiku and v10's Sonnet (v7's
    rubric), v16's Haiku and Sonnet (E3), v21's one-shot Opus (both rubrics): same label share and
    the paired difference in the real neurons' and pools' A share.
S6. Cost (CLI-reported), input, output and thinking tokens, and resolved model ids, per series.

Exploratory, listed now: how many patterns one-(main-)job answers list for each group; three
examples per series of random-text units called one (main) job, with their stated patterns.
Anything else in the analysis is labeled post hoc.
"""

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import MODELS

READERS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
V7_RUBRIC, E3 = "v7", "E3"
RUBRICS = (V7_RUBRIC, E3)
ARMS = {f"{reader}-{rubric}": (model, rubric) for reader, model in READERS.items() for rubric in RUBRICS}
RANDOM, REAL, POOL = "random", "real", "pooled 50/50"
GROUPS = (RANDOM, REAL, POOL)
ONE_JOB, RELATED, C_LABEL = "A", "B", "C"

DEPTH = "middle"
N_PER_GROUP = 100
RANDOM_UNIT_SIZE = 50
V20_RANDOM_SEED = (SEED, 20, 2)  # v20's rng_for(model, purpose): [SEED, RANK_SEED, 2, size index, purpose]
V20_RANDOM_PURPOSE = 101
SEED_OFFSET = 24
RETEST_PER_GROUP = 10
WORKERS = 6
SPEND_CAP_USD = 30.0
BOOTSTRAP = 2000
FLOOR_SIZABLE = 0.10

PILOT_MODEL, PILOT_DEPTH = "160m", "quarter"
PILOT_RANDOM_PURPOSE = 241
PILOT_UNITS = (RANDOM, REAL)

SIZABLE_FLOOR = "sizable floor"
NO_SIZABLE_FLOOR = "no sizable floor"
UNCLEAR_FLOOR = "unclear floor"
REAL_LIKE_POOLS = "real neurons like known pools"
REAL_BEYOND_POOLS = "real neurons beyond known pools"
REAL_BETWEEN = "real neurons between random text and pools"
REAL_BELOW_RANDOM = "real neurons below random text"
UNDEFINED_POSITION = "undefined: pools look like random text"
UNCLEAR_POSITION = "unclear"


def per_size_counts(n: int = N_PER_GROUP) -> dict[str, int]:
    """n units spread over the sizes as v20 spreads its random units: the first n % 6 sizes get one more."""
    return {size: n // len(MODELS) + (i < n % len(MODELS)) for i, size in enumerate(MODELS)}


def random_rng(size: str, purpose: int = V20_RANDOM_PURPOSE):
    """v20's generator for its random-text units of one size."""
    return np.random.default_rng([*V20_RANDOM_SEED, MODELS.index(size), purpose])


def random_unit_keys(positions: np.ndarray, n_units: int, rng) -> list[np.ndarray]:
    """n_units units of RANDOM_UNIT_SIZE positions each, drawn without replacement within a unit, sorted, as v20 draws them."""
    return [positions[np.sort(rng.choice(len(positions), size=RANDOM_UNIT_SIZE, replace=False))] for _ in range(n_units)]


def anchor_ids(ids: list[str], size: str, group: str, n: int) -> list[str]:
    """A seeded draw of n of one size's units of a group (v7's real neurons or 50/50 pools), in v7's order."""
    rng = np.random.default_rng([SEED, SEED_OFFSET, MODELS.index(size), GROUPS.index(group)])
    return [ids[i] for i in sorted(rng.choice(len(ids), size=n, replace=False))]


def retest_ids(ids_by_group: dict[str, list[str]]) -> list[str]:
    """RETEST_PER_GROUP units of each group, a seeded draw, the same for every series."""
    rng = np.random.default_rng([SEED, SEED_OFFSET, 9])
    return [ids_by_group[g][i] for g in GROUPS
            for i in sorted(rng.choice(len(ids_by_group[g]), size=RETEST_PER_GROUP, replace=False))]


def reading_order(n_jobs: int, purpose: int = 0) -> list[int]:
    """A seeded random order of n_jobs readings (first readings with purpose 0, retests with 1)."""
    return [int(i) for i in np.random.default_rng([SEED, SEED_OFFSET, 100 + purpose]).permutation(n_jobs)]


def floor_verdict(share_interval: list[float]) -> str:
    """P1's reading of the random-text one-(main-)job share's interval."""
    if share_interval[0] >= FLOOR_SIZABLE:
        return SIZABLE_FLOOR
    if share_interval[1] < FLOOR_SIZABLE:
        return NO_SIZABLE_FLOOR
    return UNCLEAR_FLOOR


def position(real: float, random: float, pools: float) -> float:
    """P5: where real neurons sit on the line from random text (0) to known pools (1)."""
    return (real - random) / (pools - random)


def position_reading(denominator_interval: list[float], p_interval: list[float] | None) -> str:
    """P5's reading from the interval of pools - random and, if defined, of p."""
    if denominator_interval[0] <= 0 <= denominator_interval[1] or p_interval is None:
        return UNDEFINED_POSITION
    low, high = p_interval
    if low <= 1 <= high:
        return REAL_LIKE_POOLS
    if low > 1:
        return REAL_BEYOND_POOLS
    if low >= 0 and high < 1:
        return REAL_BETWEEN
    if high < 0:
        return REAL_BELOW_RANDOM
    return UNCLEAR_POSITION
