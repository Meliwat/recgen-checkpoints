"""Pre-registered v7 rules, written and committed before any language-model reading.

v5 and v6 found that how mixed a neuron looks depends on who reads its strongest examples: a
sentence model (MiniLM) and the network's own representation give different scale trends, and
MiniLM barely tells two pooled neurons from one. The essay's own cleanliness numbers come from a
third kind of reader, a language model reading the examples. v7 puts all three readers on the
same units, including synthetic units with known answers, at the middle layer of the six sizes.
See README, "v7".

Units, per model (middle layer, v5/v6's sampled neurons):
- real: N_REAL sampled neurons, their 50 strongest examples.
- pooled: two random sampled neurons' examples, POOL_SIZE in all, MIXES[mix] from the first and
  the rest from the second, skipping examples the first already contributed (N_POOLED per mix).
- related: like a 50/50 pool, but the two neurons are the most alike by input direction
  (cosine of input weights times LayerNorm gain), paired greedily without reuse (N_RELATED).
  Their input directions make them related by the network's own reading, whatever the text says.

Readers, each giving a yes/no "unrelated mixed" flag per unit:
- MiniLM: v5's detector (recgen/analyze_v6.MiniLM).
- network: v6's detector at its pre-registered settings, with every pooled neuron's input
  direction projected out (recgen/analyze_v6.Layer).
- LLM: LLM_MODEL through the claude CLI, with no tools and only SYSTEM_PROMPT. It sees the unit's
  examples in a seeded random order, with the firing token marked and no activation values, and
  answers A (one job), B (related jobs) or C (unrelated jobs). C is the flag. Answers are cached,
  and RETEST units are asked a second time to measure how stable the labels are.

Pre-registered questions, per reader:
1. Power: the 50/50 pooled flag rate exceeds the real flag rate by at least POWER_MARGIN, with
   the pooled interval's low end above the real interval's high end, in at least MIN_MODELS models.
2. Specificity: the related-pool flag rate is at least POWER_MARGIN below the 50/50 pooled rate
   in at least MIN_MODELS models.
3. Agreement on real neurons: Cohen's kappa between each pair of readers, per model.
4. Scale: v5's verdict rule on each reader's real flag rate across the six sizes.
"""

import json
import re

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import MODELS, bootstrap_interval

DEPTH = "middle"
N_REAL = 100
N_POOLED = 50
N_RELATED = 50
POOL_SIZE = 50
MIXES = {"50/50": 25, "70/30": 35, "90/10": 45}  # examples from the first neuron
RETEST = 60
POWER_MARGIN = 0.20
MIN_MODELS = 5

LLM_MODEL = "haiku"
ONE_JOB, RELATED_JOBS, UNRELATED_JOBS = "A", "B", "C"
SNIPPET_BEFORE, SNIPPET_AFTER = 24, 4  # tokens of context shown around the marked token
SYSTEM_PROMPT = "You analyze units in a language model. Reply with JSON only."
INSTRUCTIONS = """Below are {n} text snippets where one unit in a language model fires strongly. In each, the token where it fires is marked [[like this]].

Decide which describes what the unit responds to:
A) ONE JOB: all or nearly all snippets share one pattern. The pattern may be broad or abstract, such as a grammatical role, a spelling fragment, or a kind of context.
B) RELATED JOBS: two or more patterns that are clearly related to each other.
C) UNRELATED JOBS: two or more patterns with no clear relation to each other.

Reply with JSON only, in this form: {{"label": "A", "patterns": ["short description", ...]}}

Snippets:
{snippets}"""


def rng_for(model: str, purpose: int):
    return np.random.default_rng([SEED, 7, MODELS.index(model), purpose])


def real_units(n_sampled: int, model: str) -> np.ndarray:
    return np.sort(rng_for(model, 0).choice(n_sampled, size=min(N_REAL, n_sampled), replace=False))


def random_pairs(n_sampled: int, model: str, mix: str) -> list[tuple[int, int]]:
    rng = rng_for(model, 1 + list(MIXES).index(mix))
    return [tuple(int(x) for x in rng.choice(n_sampled, size=2, replace=False)) for _ in range(N_POOLED)]


def related_pairs(directions: np.ndarray, n: int = N_RELATED) -> list[tuple[int, int]]:
    """The n most alike pairs of input directions [S, d] by cosine, taken greedily so no neuron
    appears twice."""
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    similarity = unit @ unit.T
    i, j = np.triu_indices(len(unit), 1)
    used, pairs = set(), []
    for k in np.argsort(-similarity[i, j], kind="stable"):
        a, b = int(i[k]), int(j[k])
        if a not in used and b not in used:
            pairs.append((a, b))
            used |= {a, b}
            if len(pairs) == n:
                break
    return pairs


def pool_rows(rows_a: np.ndarray, rows_b: np.ndarray, from_a: int, size: int = POOL_SIZE) -> np.ndarray:
    """from_a of the first neuron's strongest examples, then the second's in order, skipping any the
    first already contributed, up to size."""
    chosen = list(rows_a[:from_a])
    for row in rows_b:
        if len(chosen) == size:
            break
        if row not in chosen:
            chosen.append(row)
    return np.array(chosen)


def prompt(snippets: list[str]) -> str:
    return INSTRUCTIONS.format(n=len(snippets), snippets="\n".join(f"{i + 1}. {s}" for i, s in enumerate(snippets)))


def parse_label(text: str) -> str | None:
    """The label from a reply, or None if there isn't a valid one."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            label = json.loads(match.group(0)).get("label")
        except (json.JSONDecodeError, AttributeError):
            label = None
        if label in (ONE_JOB, RELATED_JOBS, UNRELATED_JOBS):
            return label
    return None


def has_power(pooled: np.ndarray, real: np.ndarray, rng) -> tuple[bool, float]:
    """Whether a reader flags pooled units clearly more than real ones, and by how much."""
    excess = float(np.mean(pooled) - np.mean(real))
    pooled_low, _ = bootstrap_interval(np.asarray(pooled, bool), rng)
    _, real_high = bootstrap_interval(np.asarray(real, bool), rng)
    return excess >= POWER_MARGIN and pooled_low > real_high, excess


def is_specific(related: np.ndarray, pooled: np.ndarray) -> tuple[bool, float]:
    """Whether a reader flags related pools clearly less than random pools, and by how much."""
    gap = float(np.mean(pooled) - np.mean(related))
    return gap >= POWER_MARGIN, gap
