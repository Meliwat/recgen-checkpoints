"""Pre-registered v8 rules, written and committed before any agent run.

v7 found that a one-shot language-model reader (Haiku reading a unit's 50 snippets once, with no
activations or tools) is noisy and flags pooled pairs of neurons only about 21 points more often
than real neurons, while calling 42-57% of real neurons "one job". The essay's ~90% "one main job"
comes from audits that investigate each neuron. v8 asks whether an investigating agent separates
real neurons from pooled ones much better, on v7's own units. See README, "v8".

Agent. AGENT_MODEL (the model v7 used, so the comparison isolates investigation) through the
claude CLI, with no built-in tools and two tools of its own:
- examples: any ranks of the unit's ranked list of 4,096 corpus windows (one example per window,
  the token where it fires most), with activation values, so it can read weaker examples too.
- activation: the unit's activation at every token of text it writes, so it can test a pattern.
It starts from v7's prompt: the same 50 snippets in the same order, now with activation values.
It has MAX_TOOL_CALLS tool calls, then answers A, B or C as in v7, with patterns and a confidence.
No steering or ablation tool: a pool's two neurons write different output directions, so any
output-side readout would show two of them, which no real neuron can, and would reveal pooling.

Activation scale. A neuron's activation is divided by its NORMALIZING_RANK-th strongest window
maximum. A pool's activation is the larger of its two neurons' normalized activations, at every
token, whether in the corpus or in the agent's text, so a pool looks like a neuron that has both
jobs. With rank 25, a 50/50 pool's 50 strongest windows are v7's 25 + 25 examples (up to windows
the two neurons share). Every value the agent sees is on a 0-10 scale where 10 is the unit's
strongest window; for a real neuron this is the activation over its peak, whatever the normalizer.

Units, per model in MODELS: N_REAL real neurons, N_POOLED 50/50 pools of random neurons and
N_RELATED pools of related neurons, drawn with fixed seeds from v7's units (recgen/units_v7.py).
RETEST units are read a second time. v7's three readers (MiniLM, network, one-shot Haiku) are
taken from results/classification_v7.json, so all four readers are compared on the same units.

Pre-registered questions:
1. Pooling detection, per reader: the mean over models of (50/50 pool flag rate - real flag rate),
   with a bootstrap interval resampling units within each model and unit type. Flag = label C.
2. Gain from investigation (primary): the agent's excess minus one-shot Haiku's on the same units,
   with a paired bootstrap interval. gain_verdict turns it into "much better" (at least
   GAIN_MARGIN and the interval above 0), "better" (interval above 0), "worse" (interval below 0)
   or "no clear difference".
3. Secondary measure: the same two questions with B or C (not one job) as the flag.
4. One-job share of real neurons (label A), agent and one-shot Haiku, beside the essay's ~90%.
5. Specificity: related pools against 50/50 pools and against real neurons, also restricted to
   related pools without a shared window (v7 found those make the control easier).
6. Agreement on real neurons: Cohen's kappa between the agent and each v7 reader.
7. Retest: flag kappa and same-label share of the agent's two readings, with a bootstrap interval.
8. Rates by model size, descriptive only: three sizes are too few for v5's trend rule.
Exploratory, listed now: tool calls used, by unit type; flag rates by confidence; how often the
agent writes test text. A reading without a valid label after ATTEMPTS tries is left out and
counted; paired comparisons use units both readers labeled.

Before the main run, a pilot on units outside v8's sample (pythia-160m, not in MODELS) checks the
tools and estimates cost. Any change it prompts (the tool budget or the sample sizes, for cost) is
committed before the first reading of a v8 unit, and listed in the README.
"""

import json

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import MODELS as V5_MODELS
from recgen.rubric_v7 import INSTRUCTIONS as V7_INSTRUCTIONS, ONE_JOB, RELATED_JOBS, UNRELATED_JOBS

MODELS = ["70m", "410m", "2.8b"]  # smallest, middle and largest of v7's six
N_REAL = 40
N_POOLED = 40
N_RELATED = 20
RETEST = 30  # 10 per model
GAIN_MARGIN = 0.15
BOOTSTRAP = 2000
INTERVAL = 95

AGENT_MODEL = "haiku"
ATTEMPTS = 3
MAX_TOOL_CALLS = 8
EXAMPLES_PER_CALL = 20
TEXTS_PER_CALL = 5
MAX_TEXT_TOKENS = 64
NORMALIZING_RANK = 25
SCALE = 10.0
MARK_THRESHOLD = 1.0  # tokens of agent text at or above this are marked
N_WINDOWS = 4096

MUCH_BETTER, BETTER, WORSE, NO_DIFFERENCE = "much better", "better", "worse", "no clear difference"

SYSTEM_PROMPT = ("You analyze units in a language model. Investigate with the tools you are given, "
                 "then give your final answer as JSON only.")
# v7's instructions with activation values and the tools added; the A/B/C definitions are v7's.
_V7_HEAD = V7_INSTRUCTIONS.split("\n\nReply with JSON only")[0]
INSTRUCTIONS = (
    _V7_HEAD.replace(
        "is marked [[like this]].",
        "is marked [[like this]], and the number before it is the unit's activation there, on a scale where 10 "
        f"is its strongest activation over {N_WINDOWS:,} windows of text.")
    + "\n\nInvestigate before you decide. You can read more of the unit's examples, including weaker ones further "
      "down its ranked list (tool \"examples\"), and measure its activation on text you write, to test whether a "
      f"pattern really drives it (tool \"activation\"). You have {MAX_TOOL_CALLS} tool calls in all."
    + "\n\nWhen you are done, reply with JSON only, in this form: "
      '{{"label": "A", "patterns": ["short description", ...], "confidence": "high"}}, '
      "where confidence is low, medium or high.\n\nSnippets:\n{snippets}")

TOOLS = [
    {"name": "examples",
     "description": (f"The unit's examples at ranks start to start+count-1 of its ranked list: {N_WINDOWS:,} windows "
                     "of text, one example each (the token where the unit fires most in that window), strongest "
                     "first. Ranks 1 to 50 are the snippets you were shown. Returns each example's rank, activation "
                     f"and snippet, with the token marked [[like this]]. count is at most {EXAMPLES_PER_CALL}."),
     "inputSchema": {"type": "object", "properties": {
         "start": {"type": "integer", "minimum": 1, "maximum": N_WINDOWS},
         "count": {"type": "integer", "minimum": 1, "maximum": EXAMPLES_PER_CALL}}, "required": ["start", "count"]}},
    {"name": "activation",
     "description": (f"The unit's activation at every token of texts you write. Up to {TEXTS_PER_CALL} texts per call, "
                     f"each cut to {MAX_TEXT_TOKENS} tokens. Each text comes back with the tokens where the activation "
                     f"is at least {MARK_THRESHOLD:.1f} marked [[token|activation]], and its strongest token."),
     "inputSchema": {"type": "object", "properties": {
         "texts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": TEXTS_PER_CALL}},
         "required": ["texts"]}},
]
NO_CALLS_LEFT = "No tool calls left. Give your final answer now."


def rng_for(model: str, purpose: int):
    return np.random.default_rng([SEED, 8, V5_MODELS.index(model), purpose])


def choose(units: list[dict], model: str) -> list[dict]:
    """v8's units of one model, in v7's order: seeded draws from v7's real, 50/50 pooled and
    related units."""
    chosen = []
    for purpose, (kind, mix, n) in enumerate((("real", None, N_REAL), ("pooled", "50/50", N_POOLED),
                                              ("related", "50/50", N_RELATED))):
        members = [u for u in units if u["kind"] == kind and u["mix"] == mix]
        picks = rng_for(model, purpose).choice(len(members), size=n, replace=False)
        chosen += [members[i] for i in sorted(picks)]
    return chosen


def retest_ids(units: list[dict], model: str) -> list[str]:
    per_model = RETEST // len(MODELS)
    return [units[i]["id"] for i in sorted(rng_for(model, 9).choice(len(units), size=per_model, replace=False))]


def normalizers(window_max: np.ndarray) -> np.ndarray:
    """Each neuron's NORMALIZING_RANK-th strongest window maximum, from window maxima [W, k]."""
    return -np.sort(-window_max.astype(np.float64), axis=0)[NORMALIZING_RANK - 1]


def unit_values(activations: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """A unit's activation from its neurons' activations [..., k]: the largest normalized one."""
    return (activations.astype(np.float64) / norms).max(axis=-1)


def ranked_rest(window_values: np.ndarray, shown_windows) -> np.ndarray:
    """The windows not already shown, strongest first (ties by window index)."""
    rest = np.setdiff1d(np.arange(len(window_values)), np.asarray(list(shown_windows), int))
    return rest[np.argsort(-window_values[rest], kind="stable")]


def snippet_line(number: int, value: float, snippet: str) -> str:
    return f"{number}. ({value:.1f}) {snippet}"


def prompt(lines: list[str]) -> str:
    return INSTRUCTIONS.format(n=len(lines), snippets="\n".join(lines))


def parse_answer(text: str) -> dict | None:
    """The last JSON object in a reply with a valid label, as {label, patterns, confidence}, or None."""
    decoder = json.JSONDecoder()
    for start in reversed([i for i, c in enumerate(text) if c == "{"]):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("label") in (ONE_JOB, RELATED_JOBS, UNRELATED_JOBS):
            confidence = obj.get("confidence")
            return {"label": obj["label"], "patterns": [str(p) for p in obj.get("patterns") or []],
                    "confidence": confidence if confidence in ("low", "medium", "high") else None}
    return None


def excess(per_model: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Mean over models of (pooled flag rate - real flag rate), from (pooled, real) flags per model."""
    return float(np.mean([np.mean(p) - np.mean(r) for p, r in per_model]))


def paired_gain(first: list[tuple[np.ndarray, np.ndarray]], second: list[tuple[np.ndarray, np.ndarray]], rng
                ) -> tuple[float, list[float], list[float], list[float]]:
    """The first reader's excess minus the second's, on the same units, and bootstrap intervals for
    the gain and for each reader's excess. Units are resampled within each model and unit type, and
    both readers' flags move together."""
    point = excess(first) - excess(second)
    gains, a_draws, b_draws = [], [], []
    for _ in range(BOOTSTRAP):
        a, b = [], []
        for (pa, ra), (pb, rb) in zip(first, second):
            ip, ir = rng.integers(0, len(pa), len(pa)), rng.integers(0, len(ra), len(ra))
            a.append((pa[ip], ra[ir]))
            b.append((pb[ip], rb[ir]))
        a_draws.append(excess(a))
        b_draws.append(excess(b))
        gains.append(a_draws[-1] - b_draws[-1])
    tail = (100 - INTERVAL) / 2
    bounds = lambda x: [float(v) for v in np.percentile(x, [tail, 100 - tail])]  # noqa: E731
    return point, bounds(gains), bounds(a_draws), bounds(b_draws)


def gain_verdict(point: float, interval: list[float]) -> str:
    low, high = interval
    if low > 0:
        return MUCH_BETTER if point >= GAIN_MARGIN else BETTER
    if high < 0:
        return WORSE
    return NO_DIFFERENCE
