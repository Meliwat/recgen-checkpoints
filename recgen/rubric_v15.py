"""Pre-registered v15 rules, written and committed before any reading.

Question. The field's standard autointerp metric scores an explanation by prediction: an explainer
writes an explanation from a unit's top examples, and a simulator predicts the unit's activations
on held-out examples from the explanation alone (Bills et al. 2023, simulation scoring; Paulo et
al. 2024, detection scoring). Can that score tell a known mixture of two unrelated neurons (v8's
random 50/50 pools) from a real neuron? If pools score about as well as real neurons (for example
because the explainer writes a disjunctive "X or Y" explanation), the score measures how
predictable a unit is, not whether it is monosemantic. v7 and v8's label readers (one job /
related jobs / unrelated jobs) separate the same pools from real neurons with label AUCs of only
0.51 to 0.64; v15 compares the score's power with theirs. See README, "v15".

Units. Exactly v8's 300 (recgen/rubric_v8.choose on recgen/units_v7.build_units): the middle layer
of Pythia 70M, 410M and 2.8B, per size 40 real neurons, 40 random 50/50 pools and 20 related pools.
Values are on v8's 0-10 scale (recgen/probe_v8.UnitView): each neuron is divided by its 25th
strongest window maximum, a pool takes the larger of its two neurons, and 10 is the unit's
strongest window. No Pythia model is loaded: everything comes from v5's window maxima and v6's
stored residuals, as in v8.

Shown examples. The explainer sees the unit's 50 shown examples (UnitView.first, v7's examples) in
v7's order, each with its activation, exactly the snippet lines of v8's agent prompt.

Held-out set, per unit, never shown to the explainer. From the unit's ranked list of 4,096 windows
(one example per window, at the unit's strongest token there; ranks 1 to 50 are the shown
examples), PER_STRATUM windows are drawn without replacement from each stratum in STRATA: ranks 51
to 300, ranks 301 to 2,000, and the remaining ranks, 2,001 to 4,096 (the "random" stratum). The
draw and the order (all 60 shuffled together) are seeded by the unit's id, so the retest and both
simulators see the same set in the same order. Ground truth is the unit's 0-10 value there.

Prompts. All three roles go through the claude CLI with no tools (v7/v8's flags, an empty tool
list, run in an empty folder outside the repo), one shot, with v7's system prompt.
- Explainer, EXPLAINER: v8's first paragraph (snippets, marked token, 0-10 values), then a request
  for a concise explanation of what the unit responds to in at most about MAX_WORDS words, as JSON
  {"explanation": "..."}. It is not asked for a one-job or mixed label, nor told how it is scored.
- Simulator, each arm in SIMULATORS: the explanation and the 60 held-out snippets (same snippet
  format, marked token, no values), asked to predict each snippet's activation from 0 to 10 using
  only the explanation, as a strict JSON list of 60 numbers in snippet order.
- Judge, JUDGE: the explanation alone, asked whether it names two or more patterns that are
  unrelated to each other, as JSON {"unrelated_patterns": true or false}. Exploratory.
A reply that doesn't parse is retried, up to ATTEMPTS tries in all. A unit whose explainer or
simulator never gives a valid reply is left out of that arm and counted; paired comparisons use
units with a score in both. The CLI's reported cost is summed across all v15 readings (pilot
included), and no new reading starts once it reaches SPEND_CAP_USD.

Scores, per unit and simulator, over the 60 held-out examples:
- simulation score (primary): Pearson correlation of predicted with true values;
- Spearman correlation of the same;
- detection AUC: how well the predictions rank the 40 examples of the two top strata above the 20
  of the random stratum (ties count half).
A constant prediction carries no information: both correlations are then 0, and it is counted.

Primary test (simulator PRIMARY_SIMULATOR). (1) Pools minus real: the mean over the three sizes of
(mean simulation score of 50/50 pools - mean of real neurons), with a 95% bootstrap interval
(BOOTSTRAP draws, resampling units within size and unit type). (2) Score AUC: the mean over sizes of
the chance that a real neuron scores above a 50/50 pool (ties half), with an interval from the same
resampling. Verdict:
- DETECTS ("detects mixtures") if the pools-minus-real interval is entirely below 0 AND the score
  AUC's interval is entirely above 0.5;
- NO_DETECTION ("doesn't detect mixtures") otherwise, if the pools-minus-real interval lies inside
  (-EQUIVALENCE_MARGIN, +EQUIVALENCE_MARGIN), so pools score about as well as real neurons;
- INCONCLUSIVE otherwise.
Descriptive, not part of the verdict: the score AUC beside v8's label AUCs (read from
results/classification_v8.json; 0.51 to 0.64), whether its interval lies above v8's largest, and
whether its point estimate reaches SCREEN_AUC, a level at which it would be a usable screen.

Secondary:
1. The same test, with its verdict, for the other simulator.
2. Related pools: related minus real and 50/50 pools minus related, and the AUC of real over
   related, for both simulators.
3. Spearman and detection AUC in place of the simulation score, for both simulators.
4. Coverage of 50/50 pools. Each held-out example is driven by the pool neuron with the larger
   normalized activation in its window. A pool's majority neuron drives more of its 40 top-strata
   held-out examples (ties go to the pool's first neuron); the other is its minority neuron. For each
   neuron that drives at least MIN_DRIVEN top-strata examples: the detection AUC of those examples
   against the 20 random-stratum ones. Reported: mean majority and minority AUC, their paired
   difference with a bootstrap interval over pools (resampled within size), the share of pools whose
   minority AUC is below UNCOVERED_AUC, and, for reference, the share of real neurons whose detection
   AUC is below it. Pools whose minority neuron drives too few examples are counted and left out.
5. Disjunctive explanations, exploratory: the share of explanations with a DISJUNCTION_MARKERS match
   (a crude count, case-insensitive), the share the judge calls unrelated patterns, and the
   explanation's word count, by unit type; and pools' scores with and without the judge's yes.
6. Retest: RETEST_UNITS units (v8's retest units, 10 per size) get the whole pipeline again (a new
   explanation, then both simulators on it, on the same held-out set). Reported: the correlation of
   the two runs' simulation scores across units with a bootstrap interval, their mean absolute
   difference, and the same for the other simulator.
7. Exploratory, linking to v12: Spearman correlation of real neurons' simulation scores with log
   peak (v12's measure: log of the neuron's peak window maximum over the median peak of its layer's
   v6-sampled neurons), pooled over sizes and per size.
8. Descriptive: mean scores by size and unit type, the two simulators' agreement (Pearson of their
   scores across units), readings without a valid reply, retries and cost.

Pilot. Before any v15 unit is read, the whole pipeline runs on v8's six pilot units (pythia-160m,
not in the sample), kept in data/v15/pilot and results/v15_pilot_answers.json, to check prompts and
parsing. If something is broken, this file is amended with a dated, committed reason before the
first v15 unit is read.
"""

import hashlib
import json
import re

import numpy as np
from scipy.stats import spearmanr

from recgen.analyze_v8 import auc
from recgen.config import SEED
from recgen.rubric_v7 import SYSTEM_PROMPT  # noqa: F401  (v7's system prompt, for all three roles)
from recgen.rubric_v8 import BOOTSTRAP, INSTRUCTIONS as V8_INSTRUCTIONS, N_WINDOWS, RETEST, excess, paired_gain

EXPLAINER = "claude-sonnet-5-5"
SIMULATORS = {"sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
PRIMARY_SIMULATOR = "sonnet"
JUDGE = "claude-haiku-5-5"
ATTEMPTS = 3
SPEND_CAP_USD = 80.0
RETEST_UNITS = RETEST  # v8's retest units, 10 per size

N_SHOWN = 50
STRATA = {"ranks 51-300": (51, 300), "ranks 301-2000": (301, 2000), "random, ranks 2001-4096": (2001, N_WINDOWS)}
TOP_STRATA = tuple(STRATA)[:2]
RANDOM_STRATUM = tuple(STRATA)[2]
PER_STRATUM = 20
N_HELD_OUT = PER_STRATUM * len(STRATA)
MAX_WORDS = 30  # asked for, not enforced; word counts are reported

EQUIVALENCE_MARGIN = 0.10
SCREEN_AUC = 0.80
MIN_DRIVEN = 5
UNCOVERED_AUC = 0.65
DISJUNCTION_MARKERS = (r"\bor\b", r"\beither\b", r"\btwo\b", r"\bboth\b", r";", r"\balso\b", r"\bas well as\b")

DETECTS, NO_DETECTION, INCONCLUSIVE = "detects mixtures", "doesn't detect mixtures", "inconclusive"

_V8_HEAD = V8_INSTRUCTIONS.split("\n\nDecide which")[0]  # snippets, marked token, 0-10 values
EXPLAINER_INSTRUCTIONS = (
    _V8_HEAD
    + f"\n\nWrite a concise explanation of what the unit responds to, in at most about {MAX_WORDS} words."
    + '\n\nReply with JSON only, in this form: {{"explanation": "..."}}\n\nSnippets:\n{snippets}')
SIMULATOR_INSTRUCTIONS = (
    "Below is an explanation of what one unit in a language model responds to, followed by {n} text snippets. "
    "In each snippet, one token is marked [[like this]].\n\nExplanation: {explanation}\n\n"
    "Using only the explanation, predict the unit's activation at the marked token of each snippet, on a scale "
    f"from 0 to 10, where 10 is its strongest activation over {N_WINDOWS:,} windows of text and 0 means it "
    "doesn't fire.\n\nReply with JSON only: a list of {n} numbers, one per snippet, in order.\n\nSnippets:\n{snippets}")
JUDGE_INSTRUCTIONS = (
    "Here is an explanation of what one unit in a language model responds to:\n\n\"{explanation}\"\n\n"
    "Does the explanation name two or more patterns that are unrelated to each other? Reply with JSON only, "
    'in this form: {{"unrelated_patterns": true}} or {{"unrelated_patterns": false}}')


def unit_rng(unit_id: str, purpose: int):
    seed = int(hashlib.sha256(unit_id.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([SEED, 15, seed, purpose])


def held_out_ranks(unit_id: str, n_ranked: int = N_WINDOWS) -> list[tuple[int, str]]:
    """The unit's held-out examples as (rank, stratum), PER_STRATUM per stratum, shuffled together."""
    rng = unit_rng(unit_id, 0)
    picks = []
    for name, (low, high) in STRATA.items():
        ranks = rng.choice(np.arange(low, min(high, n_ranked) + 1), size=PER_STRATUM, replace=False)
        picks += [(int(r), name) for r in np.sort(ranks)]
    return [picks[i] for i in rng.permutation(len(picks))]


def explainer_prompt(lines: list[str]) -> str:
    return EXPLAINER_INSTRUCTIONS.format(n=len(lines), snippets="\n".join(lines))


def simulator_prompt(explanation: str, snippets: list[str]) -> str:
    return SIMULATOR_INSTRUCTIONS.format(n=len(snippets), explanation=explanation,
                                         snippets="\n".join(f"{i + 1}. {s}" for i, s in enumerate(snippets)))


def judge_prompt(explanation: str) -> str:
    return JUDGE_INSTRUCTIONS.format(explanation=explanation)


def _json_values(text: str, opener: str):
    """Every JSON value in text that starts at an opener character, last first."""
    decoder = json.JSONDecoder()
    for start in reversed([i for i, c in enumerate(text) if c == opener]):
        try:
            yield decoder.raw_decode(text[start:])[0]
        except json.JSONDecodeError:
            continue


def parse_explanation(text: str) -> str | None:
    for obj in _json_values(text, "{"):
        if isinstance(obj, dict) and isinstance(obj.get("explanation"), str) and obj["explanation"].strip():
            return obj["explanation"].strip()
    return None


def parse_predictions(text: str, n: int = N_HELD_OUT) -> list[float] | None:
    """A JSON list of exactly n numbers from 0 to 10, or None."""
    for obj in _json_values(text, "["):
        if (isinstance(obj, list) and len(obj) == n
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 10 for x in obj)):
            return [float(x) for x in obj]
    return None


def parse_judgment(text: str) -> bool | None:
    for obj in _json_values(text, "{"):
        if isinstance(obj, dict) and isinstance(obj.get("unrelated_patterns"), bool):
            return obj["unrelated_patterns"]
    return None


def pearson(predicted, true) -> float:
    predicted, true = np.asarray(predicted, float), np.asarray(true, float)
    if predicted.std() == 0 or true.std() == 0:
        return 0.0
    return float(np.corrcoef(predicted, true)[0, 1])


def spearman(predicted, true) -> float:
    predicted, true = np.asarray(predicted, float), np.asarray(true, float)
    if predicted.std() == 0 or true.std() == 0:
        return 0.0
    return float(spearmanr(predicted, true).statistic)


def detection_auc(predicted, strata: list[str], keep=None) -> float:
    """How well predictions rank top-strata examples (those keep accepts, if given) above random-stratum ones."""
    predicted = np.asarray(predicted, float)
    top = [i for i, s in enumerate(strata) if s in TOP_STRATA and (keep is None or keep[i])]
    rest = [i for i, s in enumerate(strata) if s == RANDOM_STRATUM]
    return auc(predicted[top], predicted[rest])


def unit_scores(predicted, true, strata: list[str]) -> dict:
    return {"pearson": pearson(predicted, true), "spearman": spearman(predicted, true),
            "detection_auc": detection_auc(predicted, strata), "constant": bool(np.std(predicted) == 0)}


def group_difference(per_model: list[tuple[np.ndarray, np.ndarray]], rng) -> tuple[float, list[float]]:
    """Mean over sizes of (first group's mean - second's), with a bootstrap interval resampling units
    within size and group."""
    _, _, interval, _ = paired_gain(per_model, per_model, rng)
    return excess(per_model), interval


def separation_auc(per_model: list[tuple[np.ndarray, np.ndarray]], rng) -> tuple[float, list[float]]:
    """Mean over sizes of the chance a unit of the first group scores above one of the second (ties
    half), with a bootstrap interval from the same resampling."""
    draws = [np.mean([auc(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]) for a, b in per_model])
             for _ in range(BOOTSTRAP)]
    return float(np.mean([auc(a, b) for a, b in per_model])), [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def verdict(difference_interval: list[float], auc_interval: list[float]) -> str:
    low, high = difference_interval
    if high < 0 and auc_interval[0] > 0.5:
        return DETECTS
    if low > -EQUIVALENCE_MARGIN and high < EQUIVALENCE_MARGIN:
        return NO_DETECTION
    return INCONCLUSIVE


def coverage(predicted, strata: list[str], drivers: list[int]) -> dict:
    """A 50/50 pool's majority and minority neurons (by top-strata held-out examples driven) and the
    detection AUC of each neuron's examples, None for a neuron driving fewer than MIN_DRIVEN."""
    top = [i for i, s in enumerate(strata) if s in TOP_STRATA]
    counts = np.bincount([drivers[i] for i in top], minlength=2)
    majority = 0 if counts[0] >= counts[1] else 1
    out = {}
    for role, neuron in (("majority", majority), ("minority", 1 - majority)):
        driven = [drivers[i] == neuron for i in range(len(strata))]
        out[f"{role} examples"] = int(counts[neuron])
        out[f"{role} auc"] = detection_auc(predicted, strata, driven) if counts[neuron] >= MIN_DRIVEN else None
    return out


def has_disjunction_marker(explanation: str) -> bool:
    return any(re.search(marker, explanation, re.IGNORECASE) for marker in DISJUNCTION_MARKERS)

