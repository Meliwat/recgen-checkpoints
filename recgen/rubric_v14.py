"""Pre-registered v14 rules, written and committed before any v14 agent reading.

v8 gave an investigating agent (the claude CLI with two tools, ranked examples and activations on
new text through Pythia, and 8 tool calls) v7's real neurons and 50/50 pools of two random neurons
(known mixtures), at the middle layer of Pythia 70M, 410M and 2.8B. Pooling detection (50/50 pool
flag rate minus real neuron flag rate, flag = C, mean over the three sizes): Haiku agent +11
points, Sonnet agent +25, one-shot Haiku (v7) +19. One-job share of real neurons: Haiku agent 66%,
Sonnet agent 20%. The essay (Cammarata 2026) reports about 90% "one main job" from strong
investigating auditors. v14 asks whether the strongest available auditor, an Opus agent, detects
pools better than the Sonnet agent, and what one-job share it gives real neurons. Only the model
changes. See README, "v14".

Arm. AGENT_MODEL, by its full id, through v8's runner (recgen/agent_v8: run_cli, read and
read_model, unchanged) with v8's CLI flags, system prompt, instructions, tools, tool budget (8
calls, with at least 4 asked for), activation scale, turn cap, $1 cap per attempt, timeout, 3
attempts and empty working folder outside the repo (all from recgen/rubric_v8 and agent_v8).
Checked on 2026-10-20 with CLI 2.1.296: both "opus" and "claude-opus-5-5" resolve to
claude-opus-5-5. No effort flag is passed and CLAUDE_CODE_EFFORT_LEVEL is unset, so the CLI uses
its default effort for the model, as v8 did (v8 passed no flag either; its environment wasn't
recorded).

Units: v8's exactly. recgen/rubric_v8.choose on v7's units gives, per size in MODELS, 40 real
neurons, 40 50/50 pools of random neurons and 20 pools of related neurons (300 in all), and
recgen/rubric_v8.retest_ids gives the 30 units read a second time. The other five readers are not
read again: MiniLM, the network reader and one-shot Haiku come from results/classification_v7.json,
the Haiku and Sonnet agents from results/v8_agent_answers.json. Every Opus prompt should be
byte-identical to v8's for the same unit; the analysis checks this against v8's prompt hashes.

Primary test: the Opus agent's pooling detection minus the Sonnet agent's (PRIMARY). Detection is
v8's excess: the mean over the three sizes of the 50/50 pool flag rate minus the real neuron flag
rate, flag = C (unrelated jobs). The difference uses v8's paired bootstrap
(recgen/rubric_v8.paired_gain: units resampled within size and unit type, both readers' flags
together, on units both labeled) and v8's verdict rule (recgen/rubric_v8.gain_verdict): "much
better" if the gain is at least 15 points (GAIN_MARGIN) and its 95% interval is above 0, "better"
if only the interval is above 0, "worse" if the interval is below 0, and otherwise "no clear
difference". With these units, v8's gain intervals were about +-13 points, so a smaller true gain
can't be told from noise.

Secondary (the same machinery; their verdicts are descriptive):
1. The Opus agent against one-shot Haiku (v7) and against the Haiku agent (v8): the same paired
   gain (SECONDARY).
2. Every reader's detection with B or C ("not one job") as the flag, and the three gains on it.
3. Label AUC: how well each reader's labels (A < B < C) rank 50/50 pools above real neurons, mean
   over sizes, with a bootstrap interval (v8's post-hoc check, listed in advance here).
4. One-job share (label A) of real neurons and of 50/50 pools, with intervals, beside the essay's
   ~90%, and the Opus agent's paired difference from the Sonnet agent on real neurons.
5. Retest: same label, same flag and flag kappa on the 30 retest units, with a bootstrap interval.
6. Agreement on real neurons: Cohen's kappa between the Opus agent and each other reader.
7. Tool use: mean tool calls, the share using at least 4, the share writing test text,
   the share reading past rank 50, mean test texts, and the cost.
8. Related pools: their flag rate against 50/50 pools and against real neurons, also restricted to
   related pools without a shared window.
9. Descriptive only: flag rates by size and unit type, flag rate by stated confidence, and v8's
   catch-all keyword count of one-job answers.

Run rules:
- Pilot: one reading of each of PILOT_UNITS, the first real neuron and the first 50/50 pool among
  v8's pilot units (pythia-160m, outside the sample), kept in data/v14/pilot, to check the CLI, the
  model id, the tools and the parsing. The rules don't change after it unless something is broken;
  any amendment is committed, with its reason, before the first v14 unit is read.
- Six readings at a time, as in v8. If Pythia runs out of memory, the run is restarted with fewer
  at a time; nothing else changes.
- Spend guard: no new reading starts once the CLI-reported cost of all v14 readings, pilot
  included, reaches SPEND_CAP_USD. If it trips, the analysis uses the readings that finished and
  says so.
- Failures: as in v8, a reading gets up to 3 tries, and one without a valid label after them
  is left out and counted. The exception is an infrastructure failure (infrastructure_failure): a
  try that ended in a CLI or API error, a usage limit, a timeout or tools that didn't start, rather
  than in the agent's own reply. Such a reading is read again from scratch on a later run, at most
  RERUNS times, as v10 did for CLI errors.
"""

AGENT_MODEL = "claude-opus-5-5"
AGENT_NAME = "opus agent"
PRIMARY = (AGENT_NAME, "sonnet agent")
SECONDARY = ((AGENT_NAME, "haiku"), (AGENT_NAME, "haiku agent"))
SPEND_CAP_USD = 150.0
PILOT_MODEL = "160m"
PILOT_UNITS = (("real", None), ("pooled", "50/50"))
RERUNS = 2

# Ends of a try that come from the agent itself: these count as v8's failed attempts.
AGENT_ERRORS = ("no valid label", "error_max_turns", "error_max_budget_usd")
# A reply without a valid label that reads like one of these is the CLI or API talking, not the agent.
INFRASTRUCTURE_PHRASES = ("usage limit", "rate limit", "hit your limit", "overloaded", "api error")


def pilot_units(units: list[dict]) -> list[dict]:
    """The first unit of each kind in PILOT_UNITS, from v8's pilot model's units in v8's order."""
    return [next(u for u in units if (u["kind"], u["mix"]) == kind) for kind in PILOT_UNITS]


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
