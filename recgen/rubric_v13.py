"""Pre-registered v13 rules, written and committed before any v13 recording.

v1 to v4 tested recursive generalization on 40 neurons of Pythia-410m layer 12 that the sentence
model (MiniLM) called mixed, with facets defined by MiniLM. v7 and v9 then showed MiniLM barely sees
mixing: its flag rate doesn't change with how mixed a synthetic unit is. So those 40 neurons may not
have been mixed in the network's own terms. v13 reruns v4's formation-time test on neurons the
network-native reader (v6) calls unrelated mixed, with the facets that reader finds.

Neurons: N_NEURONS of v6's sampled neurons at 410m's middle layer (layer 12, as in v1) that v6's
detector flags unrelated mixed, drawn with a fixed seed. Facets: v6's facets of each neuron's 50
strongest examples (clusters of 5 or more in the network-native embedding), at the final checkpoint.

Recording: v4's attribution (recgen/formation.py: direct terms of each neuron's pre-activation from
every upstream MLP neuron and attention head) at every checkpoint in rubric_v4.STEPS, on every facet
example and on v4's rule for background text.

Tests, unchanged from v4 except where noted:
- formation windows, adjacency at formation and its null (each facet against the other selected
  neurons' facets, read through this neuron's weights, window maximum), MLP drivers primary and head
  drivers beside them (rubric_v4);
- adjacency at the end: v2's rule at the final checkpoint (overlap above the 95th percentile of the
  same null), computed here from v13's own final-checkpoint recording;
- labels (rubric_v2.classify) with each facet's emergence time taken as the start of its formation
  window (v4 used v1's response threshold, which v13 doesn't compute);
- branched then separated: adjacent at formation, not at the end; counted against its chance
  expectation from the null's false-positive rate, as in v4;
- positive control: pairs adjacent at the end should mostly be adjacent at formation.
Compared with v4: 1 recursive-generalization label of 40 (0.9 by chance) and 0 of 68 unrelated pairs
branched then separated (3.1 by chance).
"""

import numpy as np

from recgen.config import SEED

MODEL = "410m"
DEPTH = "middle"
N_NEURONS = 40


def select(n_sampled: int, flagged: np.ndarray) -> np.ndarray:
    """N_NEURONS sampled-neuron indices among those flagged, with a fixed seed, sorted."""
    candidates = np.flatnonzero(flagged)
    rng = np.random.default_rng([SEED, 13])
    return np.sort(rng.choice(candidates, size=min(N_NEURONS, len(candidates)), replace=False))
