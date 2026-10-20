"""Pre-registered v18 rules, written and committed before any v18 reading.

Question. On v8's 300 units (per size of Pythia 70M, 410M and 2.8B, middle layer: 40 real neurons,
40 random 50/50 pools of two neurons, 20 related pools), every reader so far separates the known
pools from real neurons poorly: label AUCs 0.51 to 0.66 (v14's Opus agent 0.66, v8's Sonnet agent
0.64), with agents that investigated lightly (about 3 to 6 of 8 tool calls). v15's standard
simulation score (Sonnet explainer, Sonnet simulator, 60 held-out examples) gives an AUC of 0.56,
and its explanations of pools mostly capture one of the two neurons. v18 asks whether a stronger,
falsification-driven audit does better: an Opus agent that must state candidate jobs, then try to
falsify them with predictions on new text and on examples below the top 50, with twice the tool
budget. (1) Is its label better at separating pools from real neurons than v14's Opus agent (same
model, v8's light protocol)? (2) Is its final description a better predictor of held-out
activations, by v15's simulation score, than v15's one-shot explanation? See README, "v18".

Agent. AGENT_MODEL by its full id, through v8's runner (recgen/agent_v8: run_cli, read and
read_model) with v8's system prompt, CLI flags, tools (examples by rank, activation on new text),
activation scale, 3 attempts and empty working folder outside the repo. What changes:
- the prompt: v8's prompt up to and including v7's A/B/C definitions (the same 50 snippets with
  their 0-10 activations, in v7's order), then PROTOCOL, then a reply format asking for the label,
  one final description and a confidence (INSTRUCTIONS). The agent is not told that some units are
  synthetic.
- the tool budget: MAX_TOOL_CALLS calls, at least MIN_TOOL_CALLS asked for (not enforced; tool
  use is reported), enforced by v8's backend; the CLI turn cap scales with it (MAX_TURNS).
- the CLI's cost cap per attempt (BUDGET_PER_ATTEMPT_USD) and timeout (TIMEOUT_SECONDS), raised
  because a reading with twice the calls is longer. Neither is meant to bind.
- held-out windows are taken out of the examples tool. v15 scores each unit on 60 held-out windows
  (20 each from ranks 51-300, 301-2,000 and 2,001-4,096; recgen/rubric_v15.held_out_ranks), and
  PROTOCOL asks the agent to read ranks 51 to 500, so without this the agent could read held-out
  examples with their values and fit its description to them. The agent's ranked list is the
  unit's ranked list with those 60 windows removed and the rest renumbered (N_LISTED windows;
  ranks 1 to 50, the shown snippets, are unchanged), and the examples tool's description gives
  that length. The activation tool is unchanged.
No effort flag is passed, as in v8 and v14. CLI version at pre-registration: CLI_VERSION; the run
records the version it used.

Units: v8's exactly (recgen/rubric_v8.choose on recgen/units_v7.build_units) and v8's 30 retest
units (recgen/rubric_v8.retest_ids), with v8's 0-10 activation scale (recgen/probe_v8.UnitView).

Comparison readers, not read again:
- v14's Opus agent: per-unit labels from v14's results/v14_agent_answers.json. It isn't on this
  branch, so it is read (read-only) from the v14 worktree and a small extract (labels, tool use,
  cost; with the source's SHA-256) is committed here.
- v15's explanation: the Sonnet simulator's per-unit scores from results/v15_answers.json, on the
  same held-out sets.
- v8's five readers: label AUCs from results/classification_v8.json, for context only (unpaired).

Scoring the final description. v15's primary simulator (rubric_v15.SIMULATORS["sonnet"]) with
v15's simulator prompt (rubric_v15.simulator_prompt), the description in place of the explanation,
on v15's exact 60 held-out snippets of each unit, in v15's order, from results/v15_answers.json.
Parsing, retries and scores are v15's (parse_predictions, ATTEMPTS, pearson as the simulation
score). The description's length is asked for (MAX_DESCRIPTION_WORDS) and not enforced, as in v15;
word counts are reported. A unit whose agent gave no valid answer, or whose simulation never
parsed, is left out of the score comparisons and counted.

Primary test: label AUC. A reader's label is ranked A < B < C (LABEL_RANK). Per size, the AUC is the
chance a 50/50 pool ranks above a real neuron (ties half); the label AUC is its mean over the three
sizes. The test is the v18 agent's label AUC minus v14's Opus agent's, on units both labeled, with
a paired bootstrap (BOOTSTRAP draws, resampling units within size and unit type, the same draws for
both readers, as in rubric_v8.paired_gain) and a 95% percentile interval. Verdict (auc_verdict):
MUCH_BETTER if the difference is at least AUC_MARGIN and its interval is above 0; BETTER if only
the interval is above 0; WORSE if the interval is below 0; otherwise NO_DIFFERENCE.

Secondary (descriptive, with intervals):
1. Detection as in v8: mean over sizes of (50/50 pool flag rate - real flag rate), flag = C, and
   with B or C as the flag; for v18 and v14, and the paired gain v18 - v14 (rubric_v8.paired_gain,
   with rubric_v8.gain_verdict's wording).
2. One-job (A) share of real neurons and of 50/50 pools, v18 and v14, with bootstrap intervals,
   and the paired difference v18 - v14 on real neurons.
3. Simulation score of the v18 descriptions: 50/50 pools minus real (mean over sizes) and its AUC
   of real over pools, with v15's resampling (rubric_v15.group_difference, separation_auc); v15's
   verdict rule applied to them (descriptive); beside v15's AUC (0.56), with the paired difference
   of the two scores' AUCs on units both scored (same resampling, both scores drawn together).
4. Paired difference in simulation score, v18 description minus v15 explanation, per unit, on real
   neurons and on 50/50 pools separately (and related pools, descriptive): mean over sizes, with a
   bootstrap interval resampling units within size.
5. Retest: on the 30 retest units, the same-label share and the C-flag kappa with a bootstrap
   interval, as in v8, beside v14's on the same units.
6. Tool use: calls per reading, the share using at least MIN_TOOL_CALLS and all MAX_TOOL_CALLS,
   test texts per reading, the deepest rank read (as a rank in the unit's full list), the share of
   readings that read examples in PROTOCOL_RANKS; beside v14's.
7. Cost: CLI-reported, per reading and in all; tokens.
Exploratory, listed now:
- v15's tracked-neuron check (analyze_v15.tracked_neurons) on the v18 predictions for 50/50 and
  related pools, beside v15's: do the descriptions still capture one of the two neurons?
- v15's judge (rubric_v15.JUDGE, judge_prompt) on the v18 descriptions: the share naming unrelated
  patterns, by unit type, beside v15's.
- simulation score by the agent's label, for real neurons and pools; description word counts; the
  paired score difference restricted to descriptions of at most MAX_DESCRIPTION_WORDS + 5 words.
- the retest descriptions' simulation scores (Sonnet simulator) and their correlation with the
  first readings'.
- label rates and detection by size; related pools' label rates.

Run rules.
- Pilot: one reading of each of PILOT_UNITS (the first real neuron and the first 50/50 pool among
  v8's pilot units, pythia-160m, outside the sample), with the simulation of their descriptions on
  v15's pilot held-out sets, kept in data/v18/pilot and results/v18_pilot_answers.json, to check the
  protocol, the tools and the parsing, and to estimate cost. The rules change after it only if
  something is broken; any amendment is committed, with its reason, before the first v18 unit is
  read.
- Order: (1) first readings, size by size (70M, 410M, 2.8B), units in round-robin order across
  real, 50/50 and related within each size, so a spend stop leaves a balanced subset; (2) the
  simulations of their descriptions and the judge; (3) the retest readings; (4) the retest
  simulations. Readings run WORKERS at a time (v8's six); if Pythia runs out of memory, fewer, and
  nothing else changes.
- Spend: SPEND_CAP_USD of CLI-reported cost over all v18 readings, pilot included. No agent reading
  starts once that total reaches SPEND_CAP_USD - SIMULATION_RESERVE_USD, and no simulation or
  judgment once it reaches SPEND_CAP_USD. If the guard stops readings, the analysis uses what
  finished and says so.
- Failures: as in v8, a reading gets up to 3 attempts and one without a valid answer is left out
  and counted. As in v14, a reading that failed for an infrastructure reason (a CLI or API error, a
  usage limit, a timeout, tools that didn't start; infrastructure_failure) is read again from
  scratch on a later run, at most RERUNS times.
"""

import json

import numpy as np

from recgen.analyze_v8 import auc
from recgen.rubric_v7 import ONE_JOB, RELATED_JOBS, UNRELATED_JOBS
from recgen.rubric_v8 import (
    BOOTSTRAP, EXAMPLES_PER_CALL, INSTRUCTIONS as V8_INSTRUCTIONS, MARK_THRESHOLD, MAX_TEXT_TOKENS, N_WINDOWS, TEXTS_PER_CALL,
)
from recgen.rubric_v15 import N_HELD_OUT

AGENT_MODEL = "claude-opus-5-5"
AGENT_NAME = "v18 opus agent"
CLI_VERSION = "2.1.296"
MAX_TOOL_CALLS = 16
MIN_TOOL_CALLS = 8  # asked for in the prompt, not enforced
MAX_TURNS = 2 * MAX_TOOL_CALLS + 2  # v8's rule: a hard stop; the tool budget is enforced by the backend
BUDGET_PER_ATTEMPT_USD = "2"
TIMEOUT_SECONDS = 1800
MAX_DESCRIPTION_WORDS = 30
PROTOCOL_RANKS = (51, 500)
N_LISTED = N_WINDOWS - N_HELD_OUT  # the agent's ranked list, without v15's held-out windows

SPEND_CAP_USD = 120.0
SIMULATION_RESERVE_USD = 8.0
RERUNS = 2
PILOT_MODEL = "160m"
PILOT_UNITS = (("real", None), ("pooled", "50/50"))

LABEL_RANK = {ONE_JOB: 0, RELATED_JOBS: 1, UNRELATED_JOBS: 2}
AUC_MARGIN = 0.10
INTERVAL = 95
MUCH_BETTER, BETTER, WORSE, NO_DIFFERENCE = "much better", "better", "worse", "no clear difference"

_V8_HEAD = V8_INSTRUCTIONS.split("\n\nInvestigate before you decide.")[0]  # snippets, 0-10 values, A/B/C
PROTOCOL = (
    "Investigate before you decide, and try to prove your ideas wrong. Follow these steps:\n"
    "1. List one to three candidate descriptions of what the unit responds to.\n"
    "2. For each candidate, write at least 3 texts you predict will activate the unit strongly and 2 you predict "
    "won't, and measure them (tool \"activation\").\n"
    f"3. Read examples from ranks {PROTOCOL_RANKS[0]} to {PROTOCOL_RANKS[1]} of the unit's ranked list (tool "
    "\"examples\") and say whether they fit your candidates.\n"
    "4. Revise your candidates in light of what you measured and read, and test a revision if it needs it.\n"
    f"You have {MAX_TOOL_CALLS} tool calls; use at least {MIN_TOOL_CALLS} of them.")
INSTRUCTIONS = (
    _V8_HEAD + "\n\n" + PROTOCOL
    + "\n\nThen reply with JSON only, in this form: "
      '{{"label": "A", "description": "...", "confidence": "high"}}, where label is A, B or C as defined above, '
      f"description is one final description of what the unit responds to, in at most {MAX_DESCRIPTION_WORDS} words, "
      "that you would use to predict its activation on any text, and confidence is low, medium or high."
    + "\n\nSnippets:\n{snippets}")

# v8's tools; only the examples tool's list length differs (held-out windows removed).
TOOLS = [
    {"name": "examples",
     "description": (f"The unit's examples at ranks start to start+count-1 of its ranked list: {N_LISTED:,} windows "
                     "of text, one example each (the token where the unit fires most in that window), strongest "
                     "first. Ranks 1 to 50 are the snippets you were shown. Returns each example's rank, activation "
                     f"and snippet, with the token marked [[like this]]. count is at most {EXAMPLES_PER_CALL}."),
     "inputSchema": {"type": "object", "properties": {
         "start": {"type": "integer", "minimum": 1, "maximum": N_LISTED},
         "count": {"type": "integer", "minimum": 1, "maximum": EXAMPLES_PER_CALL}}, "required": ["start", "count"]}},
    {"name": "activation",
     "description": (f"The unit's activation at every token of texts you write. Up to {TEXTS_PER_CALL} texts per call, "
                     f"each cut to {MAX_TEXT_TOKENS} tokens. Each text comes back with the tokens where the activation "
                     f"is at least {MARK_THRESHOLD:.1f} marked [[token|activation]], and its strongest token."),
     "inputSchema": {"type": "object", "properties": {
         "texts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": TEXTS_PER_CALL}},
         "required": ["texts"]}},
]

# Ends of a try that come from the agent itself (v14's list): these count as v8's failed attempts.
AGENT_ERRORS = ("no valid label", "error_max_turns", "error_max_budget_usd")
INFRASTRUCTURE_PHRASES = ("usage limit", "rate limit", "hit your limit", "overloaded", "api error")


def prompt(lines: list[str]) -> str:
    return INSTRUCTIONS.format(n=len(lines), snippets="\n".join(lines))


def parse_answer(text: str) -> dict | None:
    """The last JSON object in a reply with a valid label and a non-empty description, as
    {label, description, confidence}, or None."""
    decoder = json.JSONDecoder()
    for start in reversed([i for i, c in enumerate(text) if c == "{"]):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if (isinstance(obj, dict) and obj.get("label") in LABEL_RANK and isinstance(obj.get("description"), str)
                and obj["description"].strip()):
            confidence = obj.get("confidence")
            return {"label": obj["label"], "description": obj["description"].strip(),
                    "confidence": confidence if confidence in ("low", "medium", "high") else None}
    return None


def infrastructure_failure(record: dict) -> bool:
    """Whether a reading without a label failed for a reason outside the agent in any of its tries."""
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


def round_robin(units: list[dict]) -> list[dict]:
    """Units of one size interleaved across real, 50/50 and related, each kind in v8's order."""
    kinds = [[u for u in units if u["kind"] == kind] for kind in ("real", "pooled", "related")]
    return [kind[i] for i in range(max(map(len, kinds))) for kind in kinds if i < len(kind)]


def label_auc(per_model: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mean over sizes of the chance a 50/50 pool's label ranks above a real neuron's (ties half), from
    (pool ranks, real ranks) per size."""
    return float(np.mean([auc(p, r) for p, r in per_model]))


def paired_auc_difference(first: list[tuple[np.ndarray, np.ndarray]], second: list[tuple[np.ndarray, np.ndarray]], rng
                          ) -> tuple[float, list[float], list[float], list[float]]:
    """The first reader's label AUC minus the second's on the same units (aligned arrays), and bootstrap
    intervals for the difference and for each reader's AUC. Units are resampled within size and unit
    type, and both readers' labels move together."""
    point = label_auc(first) - label_auc(second)
    differences, a_draws, b_draws = [], [], []
    for _ in range(BOOTSTRAP):
        a, b = [], []
        for (pa, ra), (pb, rb) in zip(first, second):
            ip, ir = rng.integers(0, len(pa), len(pa)), rng.integers(0, len(ra), len(ra))
            a.append((pa[ip], ra[ir]))
            b.append((pb[ip], rb[ir]))
        a_draws.append(label_auc(a))
        b_draws.append(label_auc(b))
        differences.append(a_draws[-1] - b_draws[-1])
    tail = (100 - INTERVAL) / 2
    bounds = lambda x: [float(v) for v in np.percentile(x, [tail, 100 - tail])]  # noqa: E731
    return point, bounds(differences), bounds(a_draws), bounds(b_draws)


def auc_verdict(point: float, interval: list[float]) -> str:
    low, high = interval
    if low > 0:
        return MUCH_BETTER if point >= AUC_MARGIN else BETTER
    if high < 0:
        return WORSE
    return NO_DIFFERENCE
