"""Pre-registered v19 rules, written and committed before any GPT-2 activation was recorded and
before any reading.

The repo's two methods findings come from one model family, Pythia (GPT-NeoX), whose blocks run
attention and the MLP in parallel off the same residual stream:
(a) how mixed a neuron looks depends on the reader. On pools of two random neurons' strongest
    examples (25 + 25), MiniLM (v5) flags barely more than on real neurons (v7: +3 points over the
    six sizes, interval -1 to +8), while the network-native detector (v6) and one-shot Haiku (v7)
    flag 22 and 21 points more.
(b) neurons with a larger peak activation are flagged unrelated mixed less often (v12,
    pre-registered and held out), mostly shared with how alike their top 50 examples are.
v19 asks whether both replicate in a family with a sequential residual: GPT-2 small, medium and
large. It's a replication, so v5's, v6's, v7's and v12's rules are unchanged except where GPT-2's
architecture or tokenizer forces a change, listed here. See README, "v19".

Models. MODELS (124M, 355M and 774M parameters) in float16, at the middle layer, int(0.5 * layers):
6 of 12, 12 of 24 and 18 of 36. GPT-2 XL is left out for disk space. GPT-2 was released in
float32, so as a check a float32 forward pass on the CPU over the first FP32_CHECK_WINDOWS windows
is compared with the float16 window maxima of the sampled neurons.

Corpus. The same 4,096 Pile windows, re-tokenized. data/corpus.npy holds Pythia's tokens, and a
512-token Pythia window decodes to fewer than 512 GPT-2 tokens in about 29% of windows (found by
running GPT-2's tokenizer, and nothing else of GPT-2, on 512 of the windows before this file was
committed). So a GPT-2 window is the first CONTEXT_LEN GPT-2 tokens of the Pile text that starts
where the Pythia window starts in v1's packed stream (recgen.corpus): the next STREAM_SPAN Pythia
tokens, decoded, with Pythia's EOS separators decoded as "<|endoftext|>" and so re-encoded as
GPT-2's EOS. A GPT-2 window starts at the same text and ends somewhat earlier or later.
MIN_CONTEXT (32 positions) is unchanged.

Neurons. GPT-2's block is sequential: h = x + attn(ln_1(x)), then out = h + mlp(ln_2(h)). The MLP's
activation is gelu_new after c_fc, a Conv1D whose weight is [d_model, d_mlp], so a neuron's input
weights are a column. Window maxima are recorded as in recgen/sweep.py: each neuron's largest
activation in each window, ignoring the first MIN_CONTEXT positions. v5's eligibility rule drops
near-dead neurons (peak below half the layer's median peak) and dense ones (over half the windows
at half the peak), and SAMPLE_SIZE (2,048) of the rest are drawn with v19's own seed.

Readers, rules unchanged:
- MiniLM (v5): the 32 GPT-2 tokens before the firing token and the token itself, embedded with
  all-MiniLM-L6-v2 and clustered with v1's settings (cosine distance 0.75, facets of 5 or more).
  Unrelated mixed: 2 or more facets, with the least-related pair's centroid similarity below 0.1.
- network (v6): the residual stream entering ln_2 at the example (h above, which in GPT-2 includes
  the block's own attention output), LayerNorm-normalized without gain or bias, minus its mean over
  rubric_v6's N_RANDOM random positions, with the input directions of the unit's neurons (c_fc
  column times ln_2's gain) projected out. Each unit's merge and unrelated levels are v6's frozen
  percentiles (98.2 and 81.0) of its own null, the random positions embedded the same way,
  recomputed in each GPT-2 layer. As a check, MiniLM's null is recomputed on the GPT-2 contexts of
  the random positions and the percentiles of 0.25 and 0.1 in it reported; they don't change the
  levels.
- Haiku (v7): v7's prompt, system prompt, CLI flags and labels (C, "unrelated jobs", is the flag),
  with snippets of 24 GPT-2 tokens before the marked token and 4 after, in a seeded random order,
  without activation values. Answers are cached. No new reading starts once the CLI-reported spend
  reaches SPEND_GUARD_USD; units left unread drop out of Haiku's rates and are counted.

Units per model, as in v7 (rubric_v7's constants): N_REAL (100) real neurons; N_POOLED (50) pools of
two random sampled neurons at each mix (50/50, 70/30 and 90/10 of POOL_SIZE examples); N_RELATED (50)
related pools (50/50, the two neurons with the most alike input directions, paired greedily without
reuse). RETEST units in all (20 per model, drawn from all its units) are read by Haiku a second time.

Pre-registered tests.
(a) Per reader, the over-all-models excess: the mean over the three sizes of (50/50 pooled flag rate
    minus real flag rate), with a 95% interval from EXCESS_BOOTSTRAP draws that resample each model's
    units of each kind (v11's main measure; v7's exploratory one). (a) replicates if MiniLM's
    interval includes 0 or its high end is below MINILM_MAX_EXCESS (+10 points), and the network
    reader's and Haiku's intervals each lie entirely above 0.
(b) v12's H1 and H2: for the network reader on every sampled neuron, and for Haiku on the N_REAL
    real neurons per model. Peak is log(peak / median peak of the layer's sampled neurons).
    Quartiles of peak are taken within each model over all its sampled neurons, and Haiku's real
    neurons keep those quartiles (as v12 did for v11's readings).
    H1: pooled over models, the top quartile's flag rate minus the other three's has a bootstrap
    interval (resampling neurons within models) entirely below zero, and peak's Spearman correlation
    with the flag is negative in at least MIN_MODELS_NEGATIVE models, here all three (v12 asked for
    5 of 6).
    H2: in a logistic regression of the flag on standardized log peak, peakedness, density and token
    purity with a fixed effect per model, log peak's coefficient has a bootstrap interval entirely
    below zero. Neurons with undefined peakedness (fewer than 50 firing windows) are left out of H2
    (v12's amendment) and counted.
    (b) replicates if H1 holds for the network reader. H2, and both tests on Haiku, are reported
    beside it.

Secondary, pre-registered, not part of either verdict:
- per model and unit type, each reader's flag rate with a bootstrap interval; v7's per-model power
  and specificity tests; Cohen's kappa between readers on real neurons;
- each reader's GPT-2 excess minus Pythia's (the same measure on v7's per-unit flags, middle layer
  of six sizes), with an interval from independent bootstrap draws;
- the Haiku retest: same label, same flag, and the flag's kappa with a bootstrap interval;
- v12's H1 and H2 for MiniLM on every sampled neuron, and v12's exploratory regression with the
  network-native coherence of the top 50 (v6's top-pair percentile) added, for each reader;
- checks: activations rebuilt from the stored residuals (largest error), the float32 check, MiniLM's
  null percentiles on GPT-2's contexts, the share of the residual's squared norm along its mean (v6
  point 6), the share of pools with two examples from one window, and the model the CLI resolved to.
"""

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import INTERVAL, eligible, layer_index, sample
from recgen.rubric_v7 import MIXES, N_POOLED, N_REAL
from recgen.rubric_v12 import h1 as v12_h1

MODELS = ("gpt2", "gpt2-medium", "gpt2-large")  # smallest first
HUB = "openai-community"
PARAMETERS = {"gpt2": 124e6, "gpt2-medium": 355e6, "gpt2-large": 774e6}
DEPTH = "middle"
DEPTH_FRACTION = 0.5
STREAM_SPAN = 1024  # Pythia tokens decoded from each window's start; always more than 512 GPT-2 tokens
FP32_CHECK_WINDOWS = 16
RETEST = 60
SPEND_GUARD_USD = 40.0
EXCESS_BOOTSTRAP = 2000
MINILM_MAX_EXCESS = 0.10
MIN_MODELS_NEGATIVE = 3

REPLICATES = "replicates"
DOES_NOT_REPLICATE = "does not replicate"


def middle_layer(n_layers: int) -> int:
    return layer_index(n_layers, DEPTH_FRACTION)


def rng_for(model: str, purpose: int):
    return np.random.default_rng([SEED, 19, MODELS.index(model), purpose])


def sampled_neurons(window_max: np.ndarray, model: str) -> np.ndarray:
    """v5's eligibility rule and sample, from window maxima [W, N], with v19's seed."""
    return sample(eligible(window_max), rng_for(model, 0))


def real_units(n_sampled: int, model: str) -> np.ndarray:
    """v7's draw of real neurons (indices into the model's sample), with v19's seed."""
    return np.sort(rng_for(model, 1).choice(n_sampled, size=min(N_REAL, n_sampled), replace=False))


def random_pairs(n_sampled: int, model: str, mix: str) -> list[tuple[int, int]]:
    """v7's draw of pooled pairs for one mix, with v19's seed."""
    rng = rng_for(model, 2 + list(MIXES).index(mix))
    return [tuple(int(x) for x in rng.choice(n_sampled, size=2, replace=False)) for _ in range(N_POOLED)]


def retest_indices(n_units: int, model: str) -> np.ndarray:
    """Which of a model's units Haiku reads a second time."""
    return np.sort(rng_for(model, 9).choice(n_units, size=RETEST // len(MODELS), replace=False))


def _interval(draws: np.ndarray) -> list[float]:
    tail = (100 - INTERVAL) / 2
    return [float(x) for x in np.percentile(draws, [tail, 100 - tail])]


def excess_over_models(per_model: list[tuple[np.ndarray, np.ndarray]], rng, draws: int = EXCESS_BOOTSTRAP) -> dict:
    """v11's main measure. per_model: (50/50 pooled flags, real flags) for each model. The mean over
    models of the pooled minus the real flag rate, its interval, and the bootstrap draws behind it."""
    point = float(np.mean([np.mean(p) - np.mean(r) for p, r in per_model]))
    samples = np.zeros(draws)
    for pooled, real in per_model:
        pooled, real = np.asarray(pooled, float), np.asarray(real, float)
        samples += rng.choice(pooled, (draws, len(pooled))).mean(axis=1) - rng.choice(real, (draws, len(real))).mean(axis=1)
    samples /= len(per_model)
    return {"excess": point, "interval": _interval(samples), "draws": samples}


def difference(a: dict, b: dict) -> dict:
    """a's excess minus b's, from excess_over_models results drawn independently."""
    return {"difference": a["excess"] - b["excess"], "interval": _interval(a["draws"] - b["draws"])}


def decide_a(intervals: dict[str, list[float]]) -> dict:
    """(a) from each reader's over-all-models excess interval ("minilm", "network", "llm")."""
    low, high = intervals["minilm"]
    minilm_blind = low <= 0 <= high or high < MINILM_MAX_EXCESS
    sensitive = {reader: intervals[reader][0] > 0 for reader in ("network", "llm")}
    return {"minilm_near_blind": minilm_blind, "network_above_zero": sensitive["network"],
            "llm_above_zero": sensitive["llm"],
            "verdict": REPLICATES if minilm_blind and all(sensitive.values()) else DOES_NOT_REPLICATE}


def h1(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]], rng) -> dict:
    """v12's H1 with v19's model count. groups: (log peak, quartile, flag) per model."""
    out = v12_h1(groups, rng)
    out["holds"] = out["interval"][1] < 0 and out["models_negative"] >= MIN_MODELS_NEGATIVE
    return out
