"""Pre-registered rules for v16's concurrent addendum, written and committed before any addendum reading.

Why. v16's main comparison (recgen/rubric_v16) sets each reader's E3 readings against its v7-rubric
readings from an earlier session the same night: Haiku's from v7 (read 21:21 to 22:07) and Sonnet's
from v10 (23:07 to 00:05), while v16 read from 00:54 to 04:02. Those sessions also differ in
settings: v10 kept the CLI's prompt caching and removed only CLAUDE_CODE_EFFORT_LEVEL from the CLI's
environment, while v16 turned caching off and removed every variable naming an effort level. And
Sonnet's thinking differs (seen after v16's results, so post hoc): it read with no thinking at all on
49% of v10's first readings against 12% of v16's, and within v16's E3 arm, real neurons read with no
thinking were called one main job 81% of the time against 23% with thinking. So v16's rise from v7's
rubric to E3 mixes the rubric with the session. The addendum reads both rubrics in one run, with the
same settings, so that the session is shared.

Units: v7's real neurons and 50/50 pools (recgen/units_v7.build_units, the middle layer of all six
sizes: 600 real neurons and 300 pools of two random neurons), each shown exactly as v7 shows it. No
retests and no other pools.

Arms (ARMS), two readers by two rubrics, 900 readings each, 3,600 in all:
- Rubrics: "v7", v7's exact prompt (recgen/rubric_v7.prompt, whose SHA-256 per unit matches v7's
  readings), and "E3", v16's E3 prompt (recgen/rubric_v16.prompt), byte for byte the prompt of v16's
  E3 arms. Labels are parsed as in v16 (rubric_v16.parse_label with the rubric's labels).
- Readers: haiku and sonnet, with v16's main-run settings for every reading: v7's CLI flags with the
  model swapped (llm_reader_v16.cli_for), effort variables removed and prompt caching off
  (llm_reader_v16.child_env), the CLI run in the repo folder, v7's four attempts per reading, and at
  most llm_reader_v16.WORKERS readings at once. Thinking tokens are stored with every answer.

Reading order (reading_order): all 3,600 readings, both readers and both rubrics, in one seeded random
order, so the four arms are read side by side over the same stretch of time.

Cache and spend: answers are cached in data/v16/concurrent/<arm>/llm/, keyed by the prompt's SHA-256
as in v7, and every CLI call's reported cost goes to data/v16/concurrent/spend.jsonl. Spend guard:
SPEND_CAP_USD over the addendum's CLI calls only, by the CLI's reported cost, with v16's rule (a
reading starts only if the spend so far, plus rubric_v16.RESERVE_USD for it and for each reading in
flight, stays within the cap). Expected cost (from v16): about $1.30 for Haiku and $20 for Sonnet.
Missing units are handled as in v16: left out of an arm's rates and, pairwise, out of paired
comparisons.

Statistics, intervals and the verdict rule are v16's (rubric_v16.bootstrap, rubric_v16.verdict): "over
sizes" is the mean over the six sizes, and 95% bootstrap intervals resample units within each size and
unit type, the same units for every column, so comparisons are paired by unit.

Pre-registered tests, per reader:
1. Concurrent rises: the one-main-job (A) share under E3 minus under v7's rubric, read in this run,
   for real neurons (rise_real) and 50/50 pools (rise_pool), paired by unit, over sizes; and
   rise_pool - rise_real, with v16's verdict rule.
2. The paired label AUC difference, E3 minus v7's rubric (rubric_v16.LABEL_RANK; 50/50 pools ranked
   above real neurons), over sizes.
3. Per arm, the share of readings with no thinking (zero thinking tokens) over the arm's 900 readings,
   and over its real neurons and its pools, with intervals.
4. Drift against v16's main comparison, paired by unit, for real neurons and 50/50 pools:
   - session_v7: this run's v7-rubric A share minus the earlier baseline's (v7's Haiku, v10's Sonnet);
   - session_E3: this run's E3 A share minus v16's main E3 arm's;
   - rise_gap: the main comparison's rise (v16's E3 minus the earlier baseline) minus this run's rise,
     and the same for rise_pool - rise_real.

The answer to the drift confound, per reader (drift_answer):
- REPRODUCED if this run's verdict (test 1) equals the main comparison's verdict for E3, otherwise
  NOT_REPRODUCED.
- The main comparison's rise in real neurons "depends on the session" (session_dependent) if
  rise_gap's interval for real neurons excludes 0.

Described, not tested: per arm, the one-main-job share of real neurons and of 50/50 pools among
readings with no thinking and among readings with some.
"""

from functools import partial

import numpy as np

from recgen import rubric_v7
from recgen.rubric_v16 import changes
from recgen.rubric_v16 import prompt as prompt_v16

READERS = ("haiku", "sonnet")
RUBRICS = ("v7", "E3")
ARMS = {f"{reader}-{rubric}": (reader, rubric) for reader in READERS for rubric in RUBRICS}
GROUPS = ("real", "pooled 50/50")
SPEND_CAP_USD = 40.0
SEED_OFFSET = 1602  # v16, addendum 2 (after the pilot's and the main run's orders)

REPRODUCED = "reproduced"
NOT_REPRODUCED = "not reproduced"


def make_prompt(rubric: str):
    """The function that turns a unit's snippets into the rubric's prompt."""
    return rubric_v7.prompt if rubric == "v7" else partial(prompt_v16, rubric)


def reading_order(n: int, seed: int) -> list[int]:
    """One seeded random order of all n readings, readers and rubrics mixed."""
    return [int(i) for i in np.random.default_rng([seed, SEED_OFFSET]).permutation(n)]


def drift_answer(main_verdict: str, concurrent_verdict: str, rise_gap_real: list[float]) -> dict:
    """Whether this run reproduces the main comparison's verdict, and whether the main comparison's
    rise in real neurons depends on the session (rise_gap's interval excludes 0)."""
    return {"verdict": REPRODUCED if concurrent_verdict == main_verdict else NOT_REPRODUCED,
            "session_dependent": changes(rise_gap_real)}
