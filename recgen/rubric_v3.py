"""Pre-registered v3 rules, written and committed before they were computed on real facet data.

v1 and v2 looked only at the input side: what makes a neuron fire on each facet. v3 looks at
the output side, at the final checkpoint: what the neuron's firing does for the rest of the
network on each facet. The question is which of three pictures fits a neuron whose facets
look unrelated on the input side:

- one job: the network reads the neuron the same way on both facets, and it helps on both
  (the essay's "V-shape" reading: unrelated triggers, one distinction read downstream);
- one use plus interference: it helps on one facet, and on the other its firing is harmless
  or harmful noise (the interference picture of superposition);
- two uses: it helps on both, but the network reads it differently on each.

Bigger facets. v1's facets have 5 to 26 examples, too few for per-facet tests. Each facet keeps
its examples and takes more from the neuron's pool: its strongest windows at the final
checkpoint (up to POOL_MAX, one example per window at the window's maximum, at least
POOL_FRACTION of the median of the neuron's top 50). A pool example joins the facet whose
centroid (v1's sentence embedding of its original examples) it is most similar to, if that
similarity is at least ASSIGN_MIN. Each facet takes the most similar first, up to FACET_MAX
examples including its own. Examples whose next LATER_PREDICTIONS + 1 tokens don't fit in the
window are dropped. ASSIGN_MIN and the pool were set after looking at the facet sizes they give
(input side only), before any output was computed.

Measures, per example (window w, position p, neuron n at LAYER):
- help: log-prob of the actual next token with the neuron as it is, minus with the neuron
  zero-ablated at p (only at p). Positive means the neuron helps predict that token.
- effect: the full change in next-token log-probs at p (as is minus ablated), centered over
  the vocabulary. Centering removes the softmax normalizer, so this is the change in logits.

A facet's signature is the mean effect over its examples. All facets share the neuron's
output weights, so their signatures look alike by construction. The control is a null: the
same neuron patched to a facet-typical activation (the median over its facet examples) in
N_NULL random corpus contexts, minus patched to zero there. The null's mean effect is the
neuron's generic push. A facet's specific signature is its signature minus its projection on
the generic push: what the network does with the neuron on that facet beyond what it does
anywhere. A facet's reliability is how well two random halves of it agree on their specific
signatures (mean cosine over SPLITS splits, stepped up to the full facet with Spearman-Brown:
2r / (1 + r)), a split-half ceiling. The shared-job score of facets A and B is the cosine of
their specific signatures divided by the square root of the product of their reliabilities
(the usual correction for noise): about 1 if they share one job, about 0 if their jobs are
unrelated.

Labels, per facet pair, checked in this order:
1. Either facet has fewer than MIN_EXAMPLES examples: too few examples.
2. Each facet's use: helps if its median help is at least MIN_HELP and a two-sided Wilcoxon
   signed-rank test against 0 gives p < ALPHA; harms if at most -MIN_HELP with p < ALPHA;
   otherwise harmless.
3. Both help: if either facet's reliability is below MIN_RELIABILITY, it has no reliable
   specific signature and the output side can't separate one job from two uses. Otherwise one
   job if the shared-job score is at least SAME_JOB, two uses if below.
4. One helps: one use plus harmful interference if the other harms, else plus harmless
   interference.
5. Neither helps: no use found.

A neuron's label is the label of its two least-related facets (lowest v1 centroid similarity),
the pair that got it selected. All pairs are also reported.

Reported next to the primary result, never replacing it: mean-ablation instead of zero (the
neuron's mean activation over the corpus, in the facet effects and the null), MIN_HELP of 0
and 0.03, and ASSIGN_MIN of 0.5. Exploratory: help summed over the LATER_PREDICTIONS
predictions after the immediate one (the neuron's write at p is also read at later
positions); each facet's most promoted tokens; how much of each signature is the generic push;
a positive control (facet pairs v1 and v2 both call related, such as one concept split in two)
and a negative control (the neuron patched into other traced neurons' facet contexts, which it
shouldn't share a job with: how often they reach SAME_JOB).
"""

import numpy as np
from scipy.stats import wilcoxon

POOL_FRACTION = 0.5  # of the median of the neuron's top-50 window maxima
POOL_MAX = 2000  # strongest windows considered
ASSIGN_MIN = 0.4  # sentence-model similarity to the facet's centroid
FACET_MAX = 40
MIN_EXAMPLES = 10
LATER_PREDICTIONS = 4
N_NULL = 1024
ALPHA = 0.05
MIN_HELP = 0.01  # nats
SPLITS = 50
MIN_RELIABILITY = 0.5
SAME_JOB = 0.5
SENSITIVITY_MIN_HELP = (0.0, 0.03)
SENSITIVITY_ASSIGN_MIN = 0.5

HELPS = "helps"
HARMLESS = "harmless"
HARMS = "harms"

TOO_FEW = "too few examples"
ONE_JOB = "one job"
TWO_USES = "two uses"
NOT_SPECIFIC = "both help, no reliable specific signature"
HARMFUL_INTERFERENCE = "one use + harmful interference"
HARMLESS_INTERFERENCE = "one use + harmless interference"
NO_USE = "no use found"
LABELS = [ONE_JOB, NOT_SPECIFIC, TWO_USES, HARMLESS_INTERFERENCE, HARMFUL_INTERFERENCE, NO_USE, TOO_FEW]


def scorable(pos: int, context_len: int) -> bool:
    """Whether the next LATER_PREDICTIONS + 1 tokens after pos are inside the window."""
    return pos + LATER_PREDICTIONS + 1 < context_len


def enlarge(similarities: np.ndarray, originals: list[list[int]], keep: np.ndarray,
            assign_min: float = ASSIGN_MIN, facet_max: int = FACET_MAX) -> list[list[int]]:
    """Pool rows in each enlarged facet: its original rows that are kept, then the most similar
    other kept rows whose nearest facet it is, at assign_min or above, up to facet_max in all.
    similarities[i, f] is pool row i's similarity to facet f's centroid."""
    taken = {i for rows in originals for i in rows}
    nearest = similarities.argmax(axis=1)
    facets = []
    for f, rows in enumerate(originals):
        own = [i for i in rows if keep[i]]
        extra = [i for i in np.argsort(-similarities[:, f], kind="stable")
                 if i not in taken and keep[i] and nearest[i] == f and similarities[i, f] >= assign_min]
        facets.append(own + [int(i) for i in extra[:max(0, facet_max - len(own))]])
    return facets


def use(helps: np.ndarray, min_help: float = MIN_HELP) -> str:
    """Whether the neuron helps, harms or does neither on a facet, from per-example help."""
    if np.all(helps == 0):
        return HARMLESS
    median, p = float(np.median(helps)), wilcoxon(helps).pvalue
    if p < ALPHA and median >= min_help:
        return HELPS
    if p < ALPHA and median <= -min_help:
        return HARMS
    return HARMLESS


class Signatures:
    """Facet signatures from dot products only, so any subset's mean can be compared.
    gram[i, j]: dot product of examples i and j's effects; generic[i]: example i's effect
    dotted with the generic push; generic_norm2: the generic push's squared norm."""

    def __init__(self, gram: np.ndarray, generic: np.ndarray, generic_norm2: float):
        self.gram, self.generic, self.generic_norm2 = gram, generic, generic_norm2

    def _dot(self, s, t) -> float:
        return float(self.gram[np.ix_(s, t)].mean())

    def specific_dot(self, s, t) -> float:
        """Dot product of the specific signatures (generic push projected out) of subsets s and t."""
        return self._dot(s, t) - float(self.generic[s].mean() * self.generic[t].mean()) / self.generic_norm2

    def specific_cosine(self, s, t) -> float:
        return self.specific_dot(s, t) / np.sqrt(self.specific_dot(s, s) * self.specific_dot(t, t))

    def generic_share(self, s) -> float:
        """Share of the signature's squared norm that lies along the generic push."""
        return float(self.generic[s].mean() ** 2 / self.generic_norm2 / self._dot(s, s))

    def raw_cosine(self, s, t) -> float:
        return self._dot(s, t) / np.sqrt(self._dot(s, s) * self._dot(t, t))

    def reliability(self, members, rng, splits: int = SPLITS) -> float:
        """Split-half agreement of the specific signature, stepped up to the full facet."""
        r = np.mean([self.specific_cosine(order[:len(order) // 2], order[len(order) // 2:])
                     for order in (rng.permutation(members) for _ in range(splits))])
        return spearman_brown(r)


def spearman_brown(half_agreement: float) -> float:
    return max(0.0, 2 * half_agreement / (1 + half_agreement))


def shared_job(specific_cosine: float, reliability_a: float, reliability_b: float) -> float:
    return float(specific_cosine / np.sqrt(reliability_a * reliability_b))


def pair_label(n_a: int, n_b: int, use_a: str, use_b: str, reliability_a: float, reliability_b: float,
               score: float) -> str:
    if min(n_a, n_b) < MIN_EXAMPLES:
        return TOO_FEW
    uses = {use_a, use_b}
    if uses == {HELPS}:
        if min(reliability_a, reliability_b) < MIN_RELIABILITY:
            return NOT_SPECIFIC
        return ONE_JOB if score >= SAME_JOB else TWO_USES
    if HELPS in uses:
        return HARMFUL_INTERFERENCE if HARMS in uses else HARMLESS_INTERFERENCE
    return NO_USE


def least_related_pair(facet_similarities: dict[str, float]) -> tuple[int, int]:
    """The facet pair with the lowest v1 centroid similarity, keyed "i-j"."""
    key = min(facet_similarities, key=facet_similarities.get)
    i, j = key.split("-")
    return int(i), int(j)
