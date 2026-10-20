"""Pre-registered v22 rules, written and committed before any GPT-2 activation was recorded on the new corpus and
before any reading.

v19 replicated two of the repo's methods findings on GPT-2 small, medium and large (middle layer):
(a) MiniLM barely tells pools of two random neurons' strongest examples from real neurons (-2 points over the three
    sizes), while the network reader (+22; +35 with token position projected out, post hoc) and one-shot Haiku (+22)
    do; and
(b) the network reader flags neurons with a larger peak activation unrelated mixed less often (Haiku pointed the same
    way but didn't pass).
An open objection: v19's corpus is Pile text, Pythia's training data, which is off-distribution for GPT-2 (trained on
WebText). And Haiku flagged far fewer real GPT-2 neurons (8, 18 and 25% by size) than Pythia neurons (v7: 33 to 47%).
v22 asks whether v19's results hold, and whether Haiku's flag rate on real neurons changes, when the same GPT-2
neurons are read on WebText-like text. See README, "v22".

Corpus. CORPUS_REPO on the HuggingFace hub, the first 10,000 documents of the OpenWebText corpus (Gokaslan and Cohen
2019, an open replication of WebText: pages linked from Reddit posts with at least 3 karma), published by Stas
Bekman as a test subset. Its loading script is never run: the hub's own parquet conversion of it (CORPUS_REVISION,
CORPUS_FILE) is downloaded into its own directory and checked against CORPUS_SHA256. The full corpus's dataset card
on the hub (Skylion007/openwebtext) gives the license as CC0-1.0; the subset's card names none, and the pages' text
belongs to their authors. OpenWebText was built like WebText, so some of its pages are likely in GPT-2's training
data; that is what "WebText-like" means here, and it is the point.
Windows are built with v1's construction (recgen.corpus), with GPT-2's tokenizer and nothing re-tokenized: the
documents in the file's order, each followed by GPT-2's end-of-text token, packed into one stream, cut into full
CONTEXT_LEN (512) token windows, of which N_WINDOWS (4,096) are drawn with recgen.corpus.window_starts (seeded with
SEED, kept in stream order). If the sample held fewer than N_WINDOWS full windows, all of them would be used, and
v22 would stop below MIN_WINDOWS. Before this file was committed the sample was downloaded and run through GPT-2's
tokenizer (and nothing else of GPT-2) to count it: 10,000 documents, 11,253,054 tokens with separators, 21,978 full
windows. So all 4,096 windows fit. v19's GPT-2 windows were the first 512 GPT-2 tokens of Pile text starting where a
Pythia window started in v1's stream; here no Pythia stream exists, so v1's construction applies directly.
MIN_CONTEXT (32 positions) is unchanged.

Models and recording, as v19 (rubric_v19, recgen.gpt2_v19): rubric_v19.MODELS in float16 on MPS, one at a time, at the middle
layer (6 of 12, 12 of 24, 18 of 36). Window maxima ignore each window's first MIN_CONTEXT positions. The residual
stream entering ln_2 is stored, LayerNorm-normalized without gain or bias, at the sampled neurons' 50 strongest
examples (one per window) and at rubric_v6's N_RANDOM random positions (the same (window, position) indices as v19,
now in OpenWebText windows). Checks: activations rebuilt from the stored residuals (largest error), and a float32
pass over the first FP32_CHECK_WINDOWS windows compared with the float16 window maxima of the sampled neurons. v19's
float32 pass ran on the CPU, which v19's post hoc precision check found computes GPT-2 medium wrongly; v22's runs on
FP32_DEVICE, the recording's device, where v19 found float32 matches a float64 reference.

Neurons: v19's sample, unchanged. Each model's 2,048 sampled neurons are read from v19's recording
(data/v19/{model}_network.npz), not redrawn, so every unit is the same neurons as in v19 with new examples. v5's
eligibility rule is recomputed on OpenWebText and reported: how many of the sampled neurons are near-dead (peak below
half the layer's median peak) or dense (over half the windows at half the peak) here, and how many of the layer's
neurons are eligible. All sampled neurons stay in every test. A neuron whose peak on OpenWebText isn't positive has
no log peak; it would be left out of (b) and counted. Neurons with undefined peakedness (fewer than 50 firing
windows) are left out of H2, as v12's amendment requires, and counted.

Readers, unchanged from v19:
- MiniLM (v5): 32 GPT-2 tokens before the firing token and the token, v1's clustering, v5's unrelated cutoff.
- network (v6): the residual above, centered on the random positions' mean, with the unit's neurons' input
  directions (c_fc column times ln_2's gain) projected out; each unit's levels are v6's frozen percentiles (98.2 and
  81.0) of its own null of random positions, recomputed on OpenWebText.
- Haiku (v7): v7's prompt, system prompt, CLI flags and labels ("haiku" through the claude CLI; C, unrelated jobs, is
  the flag), with 24 GPT-2 tokens before the marked token and 4 after, in v19's seeded order per unit id, without
  activation values. Answers are cached. No new reading starts once the CLI-reported spend, cached readings
  included, reaches SPEND_GUARD_USD; units left unread drop out and are counted.
Pre-registered secondary: the network reader with token position projected out, by recgen.position_v19's rule
(post hoc in v19): the top 2 right singular vectors of each layer's residuals regressed on ten smooth functions of
position, projected out of every example and null vector beside the neurons' directions. "position, 2 dims" is
the secondary; position_v19's 6-dimension and 6-random-direction variants are reported beside it, as in v19.

Units: v19's, by v19's seeds and code (recgen.units_v19): per model, 100 real neurons, 50 pools of two random sampled
neurons at each mix (50/50, 70/30, 90/10) and 50 related pools. Since the sample is v19's, unit ids and the neurons
in each unit are v19's. The same 60 units (20 per model) are read a second time.

Pre-registered tests.
(a) As v19: per reader, the mean over the three sizes of (50/50 pooled flag rate minus real flag rate), with v19's
    bootstrap interval. (a) holds on OpenWebText if MiniLM's interval includes 0 or its high end is below +10 points,
    and the network reader's and Haiku's intervals each lie entirely above 0 (rubric_v19.decide_a).
(b) As v19: v12's H1 and H2 for the network reader on every sampled neuron and for Haiku on the 300 real neurons,
    with peak quartiles taken within each model over its sampled neurons' OpenWebText peaks. H1 needs a negative
    Spearman correlation in all three models. (b) holds if H1 holds for the network reader.
Comparison with v19 (Pile), paired by unit, since the units are the same neurons:
(c) Haiku's real-neuron flag rate on OpenWebText minus on the Pile: the mean over the three sizes of each model's
    difference, over real neurons with a label in both readings, with an interval from PAIRED_BOOTSTRAP draws that
    resample neurons within each model, keeping each neuron's two readings together. Per model too. Reading
    (decide_c): the target is Haiku's rate on Pythia's real neurons at the sizes nearest GPT-2's (v7's 160m, 410m
    and 1b: PYTHIA_SIZE_MATCHED, mean 42%), and the gap is the target minus v19's Pile rate (17%).
    - CORPUS_MOST: the interval lies above 0 and the OpenWebText rate closes at least MOST_OF_GAP (half) of the gap.
      v19's low rate mostly reflected the corpus.
    - CORPUS_PART: the interval lies above 0, but less than half the gap is closed. The corpus is part of it.
    - NOT_CORPUS: the interval includes 0 or lies below it. v19's low rate reflects something about GPT-2's neurons
      or the setup (tokenizer, family), not the off-distribution corpus.
(d) Per reader, (a)'s excess on OpenWebText minus on the Pile (v19's per-unit flags), paired the same way: each
    model's 50/50 pools and real units are resampled with their two readings together. Reported with its interval;
    an interval excluding 0 means the corpus changed how well that reader sees pooling.

Secondary, pre-registered, not part of any verdict: everything v19 reports beside its tests (per-model rates with
intervals, v7's power and specificity tests, kappas on real neurons, the retest, MiniLM's magnitude tests, v12's
regression with coherence added, MiniLM's null percentiles on the new contexts, the residual's mean share, the share
of pools with two examples from one window, the model the CLI resolved to); the position-removed network reader
above; each reader's flag rate on real neurons and the network reader's and MiniLM's on all sampled neurons,
OpenWebText against the Pile, paired; Haiku's label shares (one job, related, unrelated) on real neurons in both
corpora; the median coherence (v6's top-pair percentile) by model in both; and, for real units, review_v19's
example checks (repeated 9-token contexts, the most common firing token's share, prompt length) on OpenWebText.
"""

import numpy as np

from recgen.config import SEED
from recgen.rubric_v5 import INTERVAL

CORPUS_REPO = "stas/openwebtext-10k"
CORPUS_REVISION = "refs/convert/parquet"
CORPUS_FILE = "plain_text/train/0000.parquet"
CORPUS_SHA256 = "757c03fd89e59a557d7768effd97c9a9383ea4a1020f8334967e692566677082"
N_DOCUMENTS = 10_000
MIN_WINDOWS = 2048
FP32_DEVICE = "mps"
SPEND_GUARD_USD = 10.0
PAIRED_BOOTSTRAP = 2000
PYTHIA_SIZE_MATCHED = ("160m", "410m", "1b")  # review_v19.SIZE_MATCHED's Pythia sizes, nearest 124M, 355M and 774M
MOST_OF_GAP = 0.5
SECONDARY_POSITION_VARIANT = "position, 2 dims"

CORPUS_MOST = "the corpus explains most of it"
CORPUS_PART = "the corpus explains part of it"
NOT_CORPUS = "not the corpus"


def rng(purpose: int):
    """Generators for v22's own statistics (v19's tests keep v19's generators)."""
    return np.random.default_rng([SEED, 22, purpose])


def _interval(draws: np.ndarray) -> list[float]:
    tail = (100 - INTERVAL) / 2
    return [float(x) for x in np.percentile(draws, [tail, 100 - tail])]


def _paired_draws(owt: np.ndarray, pile: np.ndarray, generator, draws: int) -> np.ndarray:
    """Bootstrap draws of mean(owt) - mean(pile) over units resampled with their two flags together."""
    owt, pile = np.asarray(owt, float), np.asarray(pile, float)
    assert owt.shape == pile.shape, "flags must be paired by unit"
    idx = generator.integers(0, len(owt), (draws, len(owt)))
    return owt[idx].mean(axis=1) - pile[idx].mean(axis=1)


def rate_change(per_model: list[tuple[np.ndarray, np.ndarray]], generator, draws: int = PAIRED_BOOTSTRAP) -> dict:
    """Test (c)'s measure. per_model: (OpenWebText flags, Pile flags) of the same units, in the same order, for each
    model. The mean over models of the flag rate's change, with a paired bootstrap interval."""
    point = float(np.mean([np.mean(o) - np.mean(p) for o, p in per_model]))
    samples = np.mean([_paired_draws(o, p, generator, draws) for o, p in per_model], axis=0)
    return {"change": point, "interval": _interval(samples), "draws": samples}


def excess_change(per_model: list[tuple[tuple, tuple]], generator, draws: int = PAIRED_BOOTSTRAP) -> dict:
    """Test (d)'s measure. per_model: ((50/50 pools' OpenWebText flags, their Pile flags), (real units' OpenWebText
    flags, their Pile flags)) for each model. The change in (a)'s excess, with a paired bootstrap interval."""
    pooled = rate_change([p for p, _ in per_model], generator, draws)
    real = rate_change([r for _, r in per_model], generator, draws)
    return {"change": pooled["change"] - real["change"], "interval": _interval(pooled["draws"] - real["draws"])}


def decide_c(owt_rate: float, pile_rate: float, target: float, interval: list[float]) -> dict:
    """Test (c)'s reading, from the mean real-neuron flag rates on OpenWebText and the Pile, the Pythia target, and
    the paired interval of their difference."""
    assert target > pile_rate, "the reading assumes Haiku's Pile rate is below Pythia's"
    closed = (owt_rate - pile_rate) / (target - pile_rate)
    if interval[0] <= 0:
        verdict = NOT_CORPUS
    else:
        verdict = CORPUS_MOST if closed >= MOST_OF_GAP else CORPUS_PART
    return {"share_of_gap_closed": float(closed), "verdict": verdict}
