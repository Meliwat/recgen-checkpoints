"""Pre-registered v21 rules, written and committed before any v21 reading.

The essay under test (Nick Cammarata, "Neural Networks Are Already Clean", Goodfire, 2026, "The
Short Version" and footnote 1) reports a preliminary audit of Qwen3-32B neuron poles: roughly 90%
have "one main job", often with "small related offshoots", about 7% "two related jobs", and about
2 to 4% are "not yet understood". v16 gave that scheme (E3, recgen/rubric_v16) to one-shot Haiku
and Sonnet on v7's units: they called 33 to 44% of real neurons one main job, put 38 to 56% in "not
yet understood", and ranked known two-neuron pools above real neurons no better than under v7's
A/B/C rubric (label AUC 0.60 to 0.64). v14 found that an Opus agent with v8's tools calls 50% of
real neurons one job under A/B/C, with a label AUC of 0.66 for pools. v21 asks, with the strongest
reader available here (Opus):

1. Does the essay's scheme bring the one-main-job share of real neurons closer to 90%?
2. Does it then still separate known pools from real neurons?

It asks both for a one-shot reader (arm A) and for an investigating agent (arm B). Like v16, this
tests how the essay's categories behave with strong readers on known mixtures; it doesn't test the
essay's number, which comes from another model's neuron poles and a method the essay doesn't detail.
See README, "v21".

Reader: READER (claude-opus-5-5) by its full id, through the claude CLI 2.1.296, as in v14.

Arm A, one shot, concurrent (ARMS_A):
- Units: v7's real neurons and 50/50 pools of two random neurons (recgen/units_v7.build_units) at
  the middle layer of all six sizes: per size 100 real neurons and 50 pools, 900 units. Dose
  response (70/30, 90/10) and related pools are out of scope, to limit cost.
- Rubrics: v7's (rubric_v7.prompt, through llm_reader.unit_prompt: v7's exact prompt) and E3
  (rubric_v16.prompt("E3", ...): v16's exact prompt). Each unit is shown as v7 and v16 show it: the
  same 50 snippets in the same seeded order, the firing token marked, no activation values.
- CLI settings: v16's (recgen/llm_reader_v16.call_cli): v7's system prompt and flags with only the
  model changed, the CLI's prompt caching off, environment variables that name an effort level
  removed (so the CLI runs at its default effort), the CLI run in the repo folder. Thinking tokens
  are recorded with every reading, as v16 records them.
- Order (reading_order_a): one run in which both rubrics are read concurrently. The 900 units come
  in a seeded random order, each unit's two readings next to each other, which rubric first decided
  by a seeded coin flip per unit; then the retest units the same way. If the spend guard trips, the
  units read so far are a random subset with both rubrics read.
- Retest: RETEST_PER_SIZE units per size (30 in all, retest_ids_a: a seeded draw from that size's
  real neurons and 50/50 pools), each read a second time under both rubrics.
- Baselines from other runs, the same units and prompts (secondary only, since they're from other
  days): v16's Haiku and Sonnet E3 readings (results/v16_answers.json), v7's Haiku readings and
  v10's Sonnet readings under v7's rubric (as recgen/analyze_v16 loads them).

Arm B, investigating agent (ARM_B):
- v14's Opus agent setup: v8's runner (recgen/agent_v8.read_model and run_cli) and tools
  (recgen/probe_v8) with v8's CLI flags, system prompt, tools, 8 tool calls (at least 4 asked for),
  activation scale, turn cap, $1 cap per attempt, timeout, 3 attempts and empty working folder
  outside the repo, as v14 ran it. The one change is the decision paragraph of the prompt: v8's A/B/C
  definitions (rubric_v16.V7_DECISION) are replaced by E3's (rubric_v16.E3_DECISION), in
  agent_instructions. The rest of the prompt, the reply form ({"label", "patterns", "confidence"})
  and the parser (rubric_v8.parse_answer, which accepts A, B or C) are v14's. The runner checks that
  every arm B prompt, with E3's paragraph swapped back, hashes to v8's (and so v14's) prompt for the
  same unit. Effort variables are removed from the CLI's environment, as in arm A; v14 ran with
  CLAUDE_CODE_EFFORT_LEVEL unset, the only effort variable the CLI reads, so this matches v14.
  Prompt caching is left on, as in v14.
- Units: v8's 300 (rubric_v8.choose: per size 40 real neurons, 40 50/50 pools and 20 related pools,
  at the middle layer of 70M, 410M and 2.8B) and v8's 30 retest units (rubric_v8.retest_ids), the
  same units as v14.
- Comparison: v14's Opus agent readings of the same units under v8's A/B/C rubric, read only from
  v14's cache (data/v14/agent_opus/, agent_opus_retest/), used only where a cached reading's prompt
  hash equals v8's for that unit. They're from an earlier run (the same CLI version, the same day),
  so the arm B comparison is paired but not concurrent.

Labels and flags: E3 is A one main job, B two related jobs, C not yet understood; v7's rubric is A
one job, B related jobs, C unrelated jobs (rubric_v16.LABEL_NAMES). "One main job" is label A in
both. The flag is "not one main job" (any label but A). The label AUC ranks A < B < C in both
(rubric_v16.LABEL_RANK): for E3, one main job < two related < not yet understood.

Statistics (v16's): "over sizes" is the mean over sizes (six in arm A, three in arm B) of each
size's value. Intervals are 95% percentile bootstraps (rubric_v16.bootstrap, BOOTSTRAP draws) that
resample units within each size and unit group; comparisons between series are paired (each draw
resamples the same units for both) and use units both labeled.

Pre-registered tests.

Primary, arm A (one-shot Opus, v7's rubric and E3 read concurrently on the same units):
A1. The one-main-job share of real neurons under each rubric, over sizes and per size, with
    intervals; E3's against the essay's ESSAY_ONE_MAIN_JOB (rubric_v16.near_essay).
A2. v16's verdict (rubric_v16.verdict) on the paired rises, E3 minus v7's rubric, in the one-main-
    job share of real neurons (rise_real) and of 50/50 pools (rise_pool), from the intervals of
    rise_real and of rise_pool - rise_real: inflates cleanliness without discrimination, raises
    cleanliness less for known pools, no clear rise, or lowers cleanliness.
A3. The label AUC for ranking 50/50 pools above real neurons under each rubric, with intervals; the
    paired AUC difference, E3 minus v7's rubric ("changes" if its interval excludes 0,
    rubric_v16.changes); and whether E3's labels still separate pools from real neurons
    (separates: the AUC's interval lies above 0.5).

Primary, arm B (the Opus agent under E3, v8's units):
B1. Its one-main-job share of real neurons and of 50/50 pools, over sizes and per size, with
    intervals; the real neurons' share against the essay's 90%.
B2. Its label AUC (50/50 pools over real neurons), with an interval, and whether it separates.
B3. Paired with v14's Opus agent (A/B/C) on the same units: the rises in the one-main-job (one-job)
    share of real neurons and of 50/50 pools, E3 minus v14, with v16's verdict rule; and the paired
    AUC difference, E3 minus v14, "changes" if its interval excludes 0.

Secondary:
S1. Label shares (A, B, C) of real neurons and 50/50 pools in every v21 series and in v14's,
    in particular "not yet understood" under E3, beside the essay's 90 / 7 / 2 to 4%.
S2. Detection: the 50/50 pools' flag rate minus the real neurons', over sizes, with "not one main
    job" in every series and with "unrelated jobs" (C) under v7's rubric.
S3. Retest: same label, same flag and the flag's Cohen's kappa with an interval (resampling the
    units), per arm A rubric (30 units) and for arm B (30 units), beside v14's on the same 30.
S4. Arm B tool use, beside v14's on the same units: mean tool calls, share with at least 4, share
    using all 8, share writing test text, share reading past rank 50, mean test texts.
S5. Cost (CLI-reported), input, output and thinking tokens, per arm and rubric.
S6. Readers on the same units, paired but from other days: Opus E3 minus v16's Haiku E3 and Sonnet
    E3, and Opus under v7's rubric minus v7's Haiku and v10's Sonnet, in the real neurons' one-main-
    job share and the label AUC.
S7. Investigation under one rubric: the Opus agent minus one-shot Opus on v8's 240 real neurons and
    50/50 pools (which arm A also reads), over the three sizes, paired: E3 (arm B minus arm A's E3)
    and v7's rubric (v14 minus arm A's v7 rubric), in the one-main-job shares and the label AUC. The
    agent also sees activation values and is told to test patterns, so "investigation" bundles
    these, as in v8.
S8. Descriptive: per-size shares, and the related pools' one-main-job share in arm B.

Exploratory, listed now: where each unit's label moves from v7's rubric to E3 (arm A, concurrent;
arm B against v14); patterns listed per one-main-job answer; v8's crude catch-all keyword count in
one-main-job answers (recgen/analyze_v8.CATCH_ALL_WORDS); arm B's labels by stated confidence.
Anything else in the analysis is labeled post hoc.

Run rules:
- Pilot, before any v21 unit: PILOT_READINGS_A, two arm A readings of v16's pilot units (pythia-160m
  at the quarter depth, outside v7's sample): the first real neuron under v7's rubric and the 50/50
  pool under E3; and two arm B readings, of v14's pilot units (the first real neuron and the first
  50/50 pool among v8's pythia-160m units) under arm B's prompt. They check the CLI, the model id,
  the tools and the parsing, and give a cost estimate. The rules change only if something is broken,
  by a committed and dated amendment, before the first v21 unit is read.
- Order: arm A first (it doesn't need the GPU, which another job may be using), then arm B. The arms
  never run at the same time, and each runs six readings at a time (at most 6 CLI calls at once).
- Spend guard: SPEND_CAP_USD of CLI-reported cost over every v21 CLI call: both pilots, both arms
  and every retry. Each arm's guard starts from the total so far. Arm A uses v16's guard (a reading
  starts only if the spend plus rubric_v16.RESERVE_USD for it and each reading in flight stays
  within the cap); arm B uses v8's (no new reading once the cap is reached). If it trips, no new
  reading starts, the analysis reports what finished, and the README says so. Expected: about $60
  for arm A (1,860 readings) and $40 for arm B (330), from v16's and v14's costs.
- Failures: arm A as in v16 (a reading that ends in a CLI error is read again on a later run; one
  without a valid label after v7's four attempts is final and missing). Arm B as in v14 (3 tries;
  a reading that failed for an infrastructure reason, infrastructure_failure, is read again from
  scratch on a later run, at most RERUNS times). Missing units are left out of that series, and
  pairwise out of paired comparisons.
- If Pythia fails to load for lack of memory, the run waits and retries; nothing else changes.
- Added 2026-10-20, while arm A ran and before any arm B unit was read, because another job downloads
  Pythia to the same disk: before each Pythia download the runner waits until at least 7 GB would
  remain free after it, polling every minute for up to an hour, and stops if it doesn't. To add this,
  the run was stopped once arm A had finished and before arm B began, and restarted. Nothing about
  what the agent sees or how readings are made changes.
"""

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import MODELS
from recgen.rubric_v8 import INSTRUCTIONS as V8_INSTRUCTIONS
from recgen.rubric_v16 import E3_DECISION, V7_DECISION

READER = "claude-opus-5-5"
V7_RUBRIC, E3 = "v7", "E3"
ARMS_A = {"opus-v7": (READER, V7_RUBRIC), "opus-E3": (READER, E3)}  # arm A series: (model, rubric)
ARM_B = "opus agent-E3"
V14_AGENT = "opus agent-v7"  # v14's Opus agent, v8's A/B/C rubric
GROUPS_A = ("real", "pooled 50/50")
RETEST_PER_SIZE = 5
ESSAY_ONE_MAIN_JOB = 0.90
CHANCE_AUC = 0.5
SPEND_CAP_USD = 140.0
SEED_OFFSET = 21
PILOT_READINGS_A = (("real", V7_RUBRIC), ("pooled 50/50", E3))  # (v16 pilot unit group, rubric)
PILOT_UNITS_B = (("real", None), ("pooled", "50/50"))  # v14's: the first of each among v8's 160m units

RERUNS = 2  # v14's
# v14's split of failed tries (recgen/rubric_v14 on branch v14): these ends come from the agent itself.
AGENT_ERRORS = ("no valid label", "error_max_turns", "error_max_budget_usd")
INFRASTRUCTURE_PHRASES = ("usage limit", "rate limit", "hit your limit", "overloaded", "api error")


def agent_instructions() -> str:
    """v8's (and v14's) agent instructions with only the decision paragraph replaced by E3's."""
    assert V8_INSTRUCTIONS.count(V7_DECISION) == 1
    return V8_INSTRUCTIONS.replace(V7_DECISION, E3_DECISION)


def agent_prompt(lines: list[str]) -> str:
    """Arm B's prompt for snippet lines numbered as v8 numbers them (rubric_v8.snippet_line)."""
    return agent_instructions().format(n=len(lines), snippets="\n".join(lines))


def as_v8_prompt(text: str) -> str:
    """An arm B prompt with E3's decision paragraph swapped back for v8's, to check it against v8's hash."""
    return text.replace(E3_DECISION, V7_DECISION)


def retest_ids_a(units: list[dict], size: str) -> list[str]:
    """Arm A's retest units of one size: a seeded draw of RETEST_PER_SIZE from its arm A units."""
    rng = np.random.default_rng([SEED, SEED_OFFSET, MODELS.index(size), 9])
    return [units[i]["id"] for i in sorted(rng.choice(len(units), size=RETEST_PER_SIZE, replace=False))]


def reading_order_a(n_units: int, seed: int, purpose: int = 0) -> list[tuple[int, str]]:
    """(unit index, rubric) for every reading of n_units units: the units in a seeded random order,
    each unit's two rubrics next to each other, which comes first decided by a seeded coin flip."""
    rng = np.random.default_rng([seed, SEED_OFFSET, purpose])
    order, v7_first = rng.permutation(n_units), rng.random(n_units) < 0.5
    pairs = {True: (V7_RUBRIC, E3), False: (E3, V7_RUBRIC)}
    return [(int(i), rubric) for i in order for rubric in pairs[bool(v7_first[i])]]


def separates(auc_interval: list[float]) -> bool:
    """Whether labels rank pools above real neurons beyond chance: the AUC's interval lies above 0.5."""
    return auc_interval[0] > CHANCE_AUC


def infrastructure_failure(record: dict) -> bool:
    """v14's rule: whether an arm B reading without a label failed, in any try, for a reason outside
    the agent (a CLI or API error, a usage limit, a timeout, tools that didn't start)."""
    if record["label"] is not None:
        return False
    for attempt in record["attempts"]:
        error = attempt["error"]
        if error is None or error == "spend cap reached":
            continue
        if error not in AGENT_ERRORS:
            return True
        reply = (attempt.get("reply") or "").lower()
        if error == "no valid label" and any(phrase in reply for phrase in INFRASTRUCTURE_PHRASES):
            return True
    return False
