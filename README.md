# Recursive generalization in Pythia's training checkpoints

Nick Cammarata's essay [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) (Goodfire, 2026-10-15) argues that most neurons which look polysemantic were never packing unrelated concepts together (superposition). Instead they grew by **recursive generalization**: a neuron starts on one pattern and sprouts locally related branches, and those branches sprout their own, so the finished neuron can end up covering things that look unrelated ("iy" → "diy" → "do it yourself").

The essay checks this by watching neurons grow in vision models it trained itself. It says the same can't be done for large language models because their training checkpoints aren't public. [Pythia](https://github.com/EleutherAI/pythia) publishes 154 checkpoints per model, so this repo runs the test on a language model.

## Setup

| | |
|---|---|
| Model | `EleutherAI/pythia-410m` (24 layers, 4,096 MLP neurons per layer, GeLU) |
| Layer | 12 (middle of the network) |
| Corpus | 4,096 random windows of 512 tokens from [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k), 2.1M tokens, from the Pile (Pythia's training data) |
| Checkpoints | steps 0, 128, 256, 512, 1k, 2k, 3k, 5k, 8k, 13k, 21k, 34k, 55k, 89k, 143k (final), log-spaced so early training is sampled densely |
| Compute | One M1 Pro laptop. About 5 minutes per checkpoint. |

I used Pythia-410m rather than OLMo-2 1B. Its checkpoints are 0.9 GB instead of 6 GB, and it has log-spaced checkpoints in early training, while OLMo's first is at step 10,000. The cost is that Pythia uses GeLU while Qwen (the essay's main language model) uses SwiGLU.

## Method

1. **Record.** For each checkpoint, run the corpus through layers 0 to 12. Store every neuron's maximum activation in each window, and every activation of the traced neurons. The first 32 positions of each window are ignored, so every example has at least 32 tokens of context.
2. **Select mixed-looking neurons, using the final checkpoint only.** Take each neuron's 50 strongest windows (one example per window). Embed 32 tokens of context up to the activating token with an independent sentence model (`all-MiniLM-L6-v2`), and cluster them (average linkage, cosine distance 0.75). Clusters with 5 or more examples are the neuron's **facets**. Near-dead neurons (peak at or below 1.0) and dense ones (half their peak in over 50% of windows) are skipped. I first tried "30% of peak in over 25% of windows", but at this layer most neurons reach 30% of peak somewhere in 80% of 512-token windows, so only 71 of 4,096 survived. I loosened it after looking at the final checkpoint and before recording any trajectory. The 40 neurons whose two least-related facets are furthest apart are traced. I also read the candidates and hand-picked the ones that look mixed to a human reader, before seeing any trajectory.
3. **Trace.** At every checkpoint, measure each facet's response: the median activation on that facet's exact final-checkpoint examples, divided by the median of the neuron's 50 strongest windows at that checkpoint.

## Rubric

Fixed in [`recgen/rubric.py`](recgen/rubric.py) and committed before any trajectory was recorded.

- A facet is **present** at a checkpoint when its response is at least 0.5.
- A facet **emerges** at the first checkpoint after which it stays present, allowing one dip. It must be present at the final checkpoint.
- Two facets are **adjacent** if either holds:
  - their embedding centroids are more similar than the 75th percentile of similarities between facets of *different* traced neurons, or
  - their most common activating tokens (lowercased, stripped) are equal, or one contains the other (2+ characters).

Each traced neuron gets one label, checked in this order:

| Label | Rule | Reading |
|---|---|---|
| unstable | some facet never emerges | can't say |
| mixed from the start | the earliest-emerging facets aren't connected by a chain of adjacent facets | against a single origin |
| packing-like | some later facet isn't adjacent to any facet that emerged before it | against recursive generalization |
| recursive generalization | every later facet is adjacent to an earlier one | consistent with the essay |
| one origin, timing unresolved | all facets emerge together and are connected | consistent, but the checkpoint grid can't show the order |

**Known bias.** Adjacency only sees relations that a sentence model or a shared token can detect. Relations the network itself might use (the essay's cat/car neuron turned out to be a "V shape" detector) are invisible here. So the rubric leans toward "packing-like" and "mixed from the start." I inspect those cases by hand afterwards, and any links found that way are reported separately; they never change a label.

## Results

### Pre-registered rubric: no recursive generalization, but the test couldn't have found it

| Label | All 40 traced | 23 hand-picked |
|---|---|---|
| recursive generalization | 0 | 0 |
| one origin, timing unresolved | 0 | 0 |
| packing-like | 35 | 19 |
| mixed from the start | 5 | 4 |
| unstable | 0 | 0 |

**Read this as a failed test, not as evidence against the essay.** Selection picked neurons whose two least-related facets have centroid similarity between -0.17 and -0.05 in the sentence model's space. The rubric then counts facets as adjacent only above 0.141 in that same space (or with a shared activating token). In a neuron whose facets are just that unrelated pair, the later facet can't be adjacent to the earlier one, so timing alone decides the label: staggered means "packing-like", simultaneous means "mixed from the start". Recursive generalization was reachable only through a bridging facet in the final top 50 or a shared token, and none turned up. Adjacency should have been judged with a different signal from the one used for selection (see [Next](#next)).

### What the trajectories show (exploratory, not pre-registered)

![Facet responses over training for the 23 hand-picked neurons](results/figures/hand_picked.png)

1. **Facets grow gradually.** None appears in a single jump. At initialization every facet response is below 0.39 (median 0.00), and 95 of the 101 facets emerge between steps 3,000 and 55,000, climbing across several checkpoints.
2. **A neuron's main job drifts in small steps.** From the strongest examples at each checkpoint ([results/trajectories.md](results/trajectories.md)):
   - 3915: "respectively" (step 512) → "though / however" (3k to 8k) → "of course / I admit" (13k) → "Yeah / Sure" (21k on)
   - 2464: "-yl" in chemical names (3k) → "volatile / variability" (8k to 13k) → "tri*ethyl* / tri*alkyl*" (21k on)
   - 1528: "want *that*" (2k to 5k) → "liked / seen" (8k) → "try / see" (13k) → "thought / want" (89k on)

   Each move is to something close by, which fits the essay's picture of local growth.
3. **Unrelated facets arrive separately, with no visible bridge.** In 33 of 40 neurons the two least-related facets emerge at different checkpoints (median gap: 2 checkpoints). Several arrive late: "Mendeleev" in 1528 at 89k, math operators in 1345 at 89k, periods in formulas in 1641 at 89k, closing quotes in dialogue in 2464 at 34k. In the 8 neurons I read checkpoint by checkpoint, none of these late facets showed up among the neuron's strongest examples on the way in, and I found no intermediate step linking them to the main job.
4. **Some neurons are repurposed rather than extended.** Neuron 3282's strongest examples go "bone" → "There / Most" → object nouns ("slows the whole *browser*", "wake up the *device*") → digits inside codes (DOIs, hex, phone numbers). Its "behind her" facet looks like a leftover from the object-noun phase. Neurons 2918 and 2382 show the same kind of replacement. The essay's own InceptionV1 example (5a:178: dome → fire engine → station wagon → dome and arch) has a similar shape.
5. **The largest facet usually doesn't come first.** If unrelated facets were offshoots of a main job, the largest facet should tend to emerge first. It did in 10 of 40 neurons (about 17 expected by chance), tied for first in 9, and came later in 21. One confound: the facet that grew most recently may dominate the final top 50, so "largest" may just mean "newest".

## What this does and doesn't show

- **It doesn't refute recursive generalization.** The pre-registered test was unable to find it. Point 3 is the closest thing here to evidence against it: in this small model, unrelated facets mostly don't arrive through a visible bridge. But that check is me reading the top 4 examples at 15 checkpoints, which can easily miss a branch growing below the top.
- **It supports one part of the essay.** A neuron's main job moves in small, readable steps (point 2).
- **It adds something the essay doesn't discuss.** Repurposing (point 4), where a neuron's main job is replaced and an old job survives as a facet, is a third way to get a mixed-looking neuron besides branching and packing.
- **Weakest part:** facets come from a sentence model. It splits one concept by genre (legal "See" became two facets of neuron 1059) and can't see relations the network might use (the essay's V shape). Both the selection and the adjacency test inherit that.
- **Scope:** one layer of one 410M-parameter model, 40 neurons, 2.1M tokens, 15 checkpoints, GeLU rather than SwiGLU.

## Next

- Judge adjacency with the network's own signal rather than the one used for selection. For example, check whether two facets share upstream neurons (attribution from layer 11), like the essay grouping a neuron's firings "by the circuits that feed them".
- Look for bridges directly. Between a facet's first rise and its emergence, measure the neuron's response to examples near both facets, not just its top examples. Pythia has a checkpoint every 1,000 steps, so the grid can be much denser.
- Repeat on more layers and on OLMo (SwiGLU), to see how common repurposing is.

## v2 results

v2 does the first two items under [Next](#next). It keeps v1's neurons, facets, emergence times and labels, and changes two things: facet adjacency is judged by the network's own upstream drivers instead of the sentence model that selected the neurons, and a bridge search looks for intermediate text at denser checkpoints. Both rules are in [`recgen/rubric_v2.py`](recgen/rubric_v2.py), committed (`76b88ed`) before either was computed on real data. No v1 file or result changed.

### v2 method

1. **Upstream drivers.** Pythia's blocks are parallel, so a layer-12 neuron's pre-activation is an exact sum of direct terms from the token embedding, each earlier attention layer, and each of the 49,152 MLP neurons in layers 0 to 11, once the LayerNorm scale is frozen at its value on each example. On the real model the terms add back up to the pre-activation within 5e-5. For each facet I average every upstream MLP neuron's term over the facet's final-checkpoint examples, subtract its average on 1,024 random text positions, and keep the 100 largest positive ones. Those are the facet's drivers.
2. **Adjacency.** Two facets are adjacent if their drivers overlap more than the 95th percentile of the neuron's null. The null is this neuron's facets against the facets of the other 39 traced neurons, all read through this neuron's weights, so sharing the neuron's own input weights can't by itself make two facets look adjacent.
3. **Bridge search.** For each later facet B and each facet A that emerged before it (69 pairs), the bridge is the 50 texts most similar to both, out of a pool of 491,520 random positions (sentence model, one text per window). I recorded pre-activations at 29 checkpoints from step 512 to 89k: v1's plus up to 3 more between each pair on Pythia's 1,000-step grid. The bridge's B-ward gain is its median pre-activation minus that of text matched on similarity to A but unlike B. A bridge **leads** if its gain passes half its level at B's emergence at least 2 checkpoints before B's own gain does, and that level is at least 20% of B's own gain then. A placebo (text matched on similarity to B but unlike A) gets the same test and should rise with B either way.

### What changed

| Label | v1, all 40 | v2, all 40 | v1, 23 hand-picked | v2, 23 hand-picked |
|---|---|---|---|---|
| recursive generalization | 0 | 2 | 0 | 0 |
| one origin, timing unresolved | 0 | 0 | 0 | 0 |
| packing-like | 35 | 32 | 19 | 19 |
| mixed from the start | 5 | 6 | 4 | 4 |
| unstable | 0 | 0 | 0 | 0 |

![Upstream driver overlap between facets, against its null and ceiling](results/figures/v2_driver_overlaps.png)

1. **The network's own drivers mostly agree with the sentence model.** Two random halves of one facet share a median 42% of their drivers, so the signature is stable enough to test. A facet and another neuron's facet share 2%. Facet pairs that v1 called unrelated also share 2%, while pairs v1 called related (such as neuron 3370's two "Sure" facets, 86%) share a median 45%. The two versions agree on 78 of 86 facet pairs. Fixing v1's circularity didn't uncover hidden adjacency: facets that look unrelated to a reader are also driven by different upstream neurons.
2. **Two neurons become recursive generalization, about what chance predicts.** If facet pairs were adjacent only at the null's false-positive rate (median 4.5% per pair), 0.9 such labels would be expected. Neuron 2976 is the interesting one: its facets are mid-word pieces of unusual words in three scripts (CJK byte fragments, "T*iss*ington" in British place names, Turkish "ç"), which looks like a single job that a sentence model can't see. Neuron 4010 (scientific vs legal citations) is weaker: two random halves of its facet 0 share only 8% of their drivers, so that facet's signature is mostly noise. Neuron 1228 moves from packing-like to mixed from the start.
3. **Sensitivity.** With top 25, 50 or 200 drivers instead of 100, 1 to 3 neurons are labeled recursive generalization. With v1's 75th percentile instead of the 95th, it's 5 to 10, but that cutoff counts a quarter of unrelated pairs as adjacent by construction.
4. **The bridge search found no leading bridge, but it had little chance to.** None of the 51 later facets has one, and 67 of 69 pairs show no bridge response at all. The placebo didn't pass either: text as similar to B as the bridge is, but unlike A, reached a median 4.8% of B's own gain (bridge: 3.7%, and above the placebo in 33 of 69 pairs, a coin flip). Text that a sentence model puts near a facet barely drives these neurons compared with the facet's own examples, so a 20% floor was out of reach for any matched text. What the search does show is that in these 29 checkpoints, nothing a sentence model sees as between two unrelated facets drove the neuron on the way. ([all pairs](results/figures/v2_bridges_all.png), [hand-picked](results/figures/v2_bridges_hand_picked.png), [gain shares](results/figures/v2_bridge_gain_shares.png))
5. **Facets arrive earlier in pre-activation than v1's threshold shows** (exploratory). When the earlier facet of a pair emerges, the later facet's examples already have a median 72% of the pre-activation gain they have at their own emergence. Timing each facet by where its gain passes half its step-89k value, v1's order holds in 49 of the 69 staggered pairs, ties in 10 and reverses in 10. Late arrivals from v1 point 3 were building for most of training: "Mendeleev" in 1528 emerges at 89k under v1 but is half-way by 17k, and 2976's facets (8k, 21k, 34k under v1) are half-way by 2k, 4k and 4k. GELU and the 0.5 cutoff make arrivals look later and more separate than the drive behind them. That weakens the timing half of both versions' labels, which decides between packing-like and mixed from the start.

### What's still shaky

- **Drivers cover a minority of the input.** Upstream MLP neurons carry a median 25% of a facet's rise over random text. The rest comes through attention and the token embedding, which v2 doesn't compare, and indirect paths (an early neuron acting through later ones) are left out.
- **Adjacency is judged at the final checkpoint.** Two facets that branched from a shared circuit and later separated would look unrelated now. Comparing drivers at the later facet's emergence checkpoint would test that directly.
- **The bridge test is underpowered** (point 4), and its texts are only weakly between facets: a median similarity of 0.24 to each, against 0.68 for a facet's own examples to its centroid. When two facets are unrelated, the 491,520-text pool holds little that is close to both.
- **The null is strict for same-genre facets.** Many traced facets come from the same few genres (legal, fiction, travel guides), so the null's upper tail includes same-genre pairs and the 95th-percentile threshold ranges from 0.05 to 0.15 by neuron.
- **Scope is unchanged:** one layer of one model, 40 neurons selected with the sentence model.

### v2 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v2.py`](recgen/rubric_v2.py) | Pre-registered adjacency and bridge rules |
| [`results/classification_v2.json`](results/classification_v2.json) | v1 and v2 labels, driver overlaps, nulls, sensitivity, bridge gains, exploratory stats |
| [`results/v2_neurons.md`](results/v2_neurons.md) | Per neuron: overlaps against the threshold, bridge outcomes, example bridge texts |
| [`results/bridges.json`](results/bridges.json) | Bridge and control sets for each facet pair |
| [`results/figures/`](results/figures/) `v2_*.png` | Driver overlaps, bridge gains, label counts |

```bash
./run_v2.sh
```

`run_v2.sh` needs v1's `results/` and `data/corpus.npy`. From scratch it takes about 2 hours on an M1 Pro: it downloads the final checkpoint once and 29 more one at a time, deleting each after use, and caches 1.6 GB in `data/`. With that cache, it reruns in a few minutes.

## v3 results

v1 and v2 looked only at the input side: what makes each neuron fire. The essay's core argument is about the output side: the cat/car neuron is one "V-shape" distinction, and the rest of the network reads it the same way whichever object set it off. v3 checks the output side of the same 40 neurons at the final checkpoint. On each facet, does the neuron's firing help predict the next token, and does the network do the same thing with it on each facet? The rules are in [`recgen/rubric_v3.py`](recgen/rubric_v3.py), committed (`815a088`) before any output was computed on facet data. No v1 or v2 file or result changed.

| Picture | What the output side should show |
|---|---|
| **one job**, unrelated triggers (the essay's V shape) | the neuron helps on both facets, and the network does the same thing with it on each |
| **one use + interference** (superposition's interference) | it helps on one facet; on the other, removing it changes nothing (harmless) or helps (harmful) |
| **two uses** sharing a neuron | it helps on both, but the network does different things with it on each |

### v3 method

1. **Bigger facets.** v1's facets have 5 to 26 examples, too few to test one at a time. Each facet keeps its examples (690 of v1's 703; the other 13 sit too near the end of a window to score) and takes more from the neuron's strongest windows at the final checkpoint (one example per window, down to half the neuron's top-50 level, at most 2,000 windows). A window joins the facet whose v1 centroid it is most similar to in the sentence model's space, if that similarity is at least 0.4, most similar first, up to 40 per facet. I set 0.4 after looking at the facet sizes it gives and reading some borderline additions (input side only), before computing any output. The 101 facets now have 7 to 40 examples (median 40), 3,127 in all.
2. **Help.** At each example, zero-ablate the neuron at that position only and compare the log-prob of the actual next token. Help is "as is" minus "ablated", in nats. A facet **helps** if its median help is at least 0.01 and a Wilcoxon signed-rank test gives p < 0.05, **harms** if the median is at most -0.01 with p < 0.05, and is **harmless** otherwise.
3. **What the network does with it.** A facet's signature is the change in next-token log-probs over the whole vocabulary, centered (so it's the change in logits), averaged over the facet's examples. All facets share the neuron's output weights, so their signatures look alike by construction. The control: patch the neuron to a typical activation (the median over all its facet examples) at 1,024 random positions in windows without v1 facet examples. The mean change there is its **generic push**. A facet's **specific signature** is its signature with the generic push projected out: what the network does with the neuron on that facet beyond what it does anywhere. Its **reliability** is how well two random halves of the facet agree on it (stepped up to the full facet with Spearman-Brown), the split-half ceiling. The **shared-job score** of two facets is the cosine of their specific signatures divided by the square root of the product of their reliabilities, the usual correction for noise: about 1 if they share one job, about 0 if their jobs are unrelated. On simulated facets with unrelated jobs, raw signatures have cosine 0.87 to 0.90, while this score stays within 0.03 of zero ([`tests/test_rubric_v3.py`](tests/test_rubric_v3.py)).
4. **Labels**, per facet pair: **too few examples** if either facet has fewer than 10. Otherwise, by each facet's help: both help gives **one job** (score 0.5 or more), **two uses** (below 0.5), or **no reliable specific signature** (either reliability below 0.5); one helps gives **one use + harmless** or **+ harmful interference**; neither helps gives **no use found**. A neuron's label is that of its two least-related facets, the pair that got it selected.

Reported next to the labels: mean-ablation instead of zero, a help floor of 0 or 0.03, and a 0.5 similarity cutoff for added examples. Exploratory: help on the 4 predictions after the next one, a positive control (facet pairs v1 and v2 both call related, such as one concept split in two), and a negative control (each neuron patched into the other neurons' facets, contexts it shouldn't share a job with).

### What v3 found

| v3 label (least-related pair) | All 40 | 23 hand-picked |
|---|---|---|
| one job | 0 | 0 |
| both help, no reliable specific signature | 0 | 0 |
| two uses | 0 | 0 |
| one use + harmless interference | 0 | 0 |
| one use + harmful interference | 0 | 0 |
| no use found | 37 | 22 |
| too few examples | 3 | 1 |

Side by side with v1 and v2 ([per neuron](results/v3_neurons.md)): all 40 are "no use found" except 3068, 138 and 2346 (packing-like in v1 and v2), whose least-related pair has a facet with under 10 examples. That covers all of v1's labels (35 packing-like, 5 mixed from the start) and both neurons v2 called recursive generalization (2976, 4010).

![Shared-job scores against the controls](results/figures/v3_shared_job.png)

1. **The pre-registered labels can't tell the three pictures apart, for two reasons.**
   - **On their strongest examples, these neurons don't measurably help predict the next token.** No facet helps; 99 of 101 are harmless and 2 harm. On the next 4 predictions, the 93 harmless facets with 10 or more examples stay harmless. Mean-ablation and a 0.03 floor give identical labels. With no floor, 5 neurons get an interference label on helps of 0.0001 to 0.007 nats, and none gets one job or two uses. Per example (exploratory), ablation moves the actual next token's log-prob by a median 0.011 nats, up or down: 49% of examples go up, the mean is 0.0001 nats, and the direction is unrelated to whether the model was right (Spearman -0.004 with its log-prob). A quarter of examples already give the next token over 90% probability, where ablation barely moves it (median 0.0002 nats, against 0.019 for the rest). ([help per neuron](results/figures/v3_primary_help.png), [labels](results/figures/v3_labels.png))
   - **Even with large help, the split between one job and two uses couldn't have caught the essay's case.** It compares final output effects. In the essay's picture, a V shape read by cat features in cat images and by car features in car images changes the output differently in each, so the rubric would have called it two uses. This is a flaw in the pre-registered design, not a finding.
2. **Where a neuron fires most, it barely changes the prediction, and no more than in random text** (exploratory, added after seeing point 1). On a facet, ablation moves a median 1.0% of next-token probability (KL 0.0004 nats). Patched to the same activation in random text, the neuron moves 1.0% (KL 0.0004). Of the 95 facets with 10 or more examples, about half are above their random-text level: 49 in KL, 46 in probability moved.
3. **What the network does with a neuron at the logits is consistent within a facet and unrelated across unrelated facets** (exploratory: the score is pre-registered, but it enters a label only when both facets help, which never happened, and both controls were declared exploratory). The raw signatures of a neuron's two least-related facets have cosine 0.39 (median): the shared-weights effect. Beyond the generic push (a median 39% of a signature's squared norm), a facet's reliability has median 0.81 (two halves agree at 0.68, stepped up to the full facet). Facet pairs v1 and v2 both call related, like 1059's two legal "See" facets, score a median 0.86, and 10 of 12 pass 0.5. The 36 least-related pairs whose facets are both reliable score a median -0.006, and none passes 0.5. That is where unrelated contexts sit: a neuron's facets against other neurons' facets, with the neuron patched in there, score a median 0.03 (95th percentile 0.36), and 2.4% of those pairs pass 0.5. The highest least-related pair, 2976 at 0.42, is about what the best of 36 such control pairs scores anyway (the maximum of 36 random control scores beats it 3 times in 4). Two limits: the least-related pairs were picked for having the least similar contexts, so a score near 0 partly follows from selection; and the positive control covers only near-identical contexts (one concept split in two), so nothing checks that the score can find one job across different contexts.

### Reading cases by hand

These readings are separate from the labels. They use the tokens whose probability the neuron raises and lowers most on each facet (listed for every neuron in [results/v3_neurons.md](results/v3_neurons.md)).

- **1059, legal "See" vs price and time ranges.** After "See" it raises the start of a cited case ("People", "United", "id", "Id") and lowers "Rule", "also", "supra" and commas: cite a case, not a rule or statute. In ranges like "5-10pm" it reshuffles which number or day comes next ("4", "9", "10", "Tue" up; "5", "6", "8", "Mon" down) with no clear direction. Two different effects on the prediction, both small (1.5% and 1.1% of probability moved).
- **3282, phone numbers vs "behind her".** In phone numbers it raises single digits and lowers "00", "0" and "-": how the number continues. After "behind her" or "in front of her" it raises ".", "," and "and" and lowers "mood", "face" and "bed": it reads "her" as an object that ends the phrase, not a possessive ("her mood"). Two readable, unrelated effects on the prediction. Its median help on "her" (+0.004 nats, p = 0.51) is indistinguishable from zero.
- **1528, "Mendeleev" vs "See you".** On "M|ende" it raises "ley", "'s" and "," and lowers "le" and "lev", the correct next piece: median help -0.03 nats, KL ten times its random-text level, the clearest harmful effect in the set. But all 21 examples come from one long text (consecutive corpus windows 1432 to 1482, no document break), so its p < 0.001 treats one book's repeated "Mendeleev" as independent examples. In dialogue it does almost nothing. This facet fits the interference picture, but v3 can't say what the neuron's main job is, since no facet helps. It isn't labeled harmful interference for the same reason.
- **2976, mid-word pieces in Chinese and Japanese, British place names, and Turkish and other languages.** On all three facets it raises plausible continuations of a word split into rare pieces ("った" and "ア" in Japanese; "inia" and "ichen" in place names; "men", "inc" and "en" in Turkish). That reads like one effect, and v2 called this neuron recursive generalization from its upstream drivers. But its least-related score (0.42) is within what unrelated pairs reach (point 3), so the output side doesn't confirm it. The "Turkish" facet is half one election table (consecutive windows 302 to 311, "Toplam Se|ç|men"), plus four other Turkish texts and Finnish, Hindi, Hungarian, Croatian and C code. Its help (median +0.05 nats, p = 0.07, 19 examples) is the largest among facets with 10 or more examples.
- Two of the next-highest scores belong to the hand-picked neurons I had flagged for a possible shared pattern, both below the negative control's 95th percentile. 2442 (commands in dialogue vs "stir" in recipes; least-related pair 1-2, score 0.28) raises what follows an imperative verb: prepositions after "stir" ("for", "until", "into") and objects or prepositions after spoken commands ("this", "for", "on"). 2918 (section signs vs pound signs, 0.23) raises digits on both facets; its pound-sign facet harms, pushing prices like "25", "45" and "15".

### What this does and doesn't show

- **It doesn't decide between the three pictures**, for the two reasons in point 1: no facet passes the help test, and the split between one job and two uses at the logits couldn't have caught the essay's case anyway. It's no evidence for or against any of the three.
- **At the logits, the network's response to a neuron follows the context.** Unrelated facets get unrelated output signatures, as unrelated contexts do. That isn't evidence against the essay, which expects the same of a V shape read by cat features in one image and car features in another. Telling "one distinction read downstream" from "two uses" needs a test one level down: whether the neuron's write feeds the same downstream components on both facets (by weights or a linearized attribution, as v2 did upstream), with the context deciding which of them act, or disjoint components on each.
- **Two cases stand out, neither decisively.** 1528 harms on "Mendeleev", but in one book. 2976 raises word continuations on all three facets, but its score is within the negative control.

### What's still shaky

- **Single-neuron ablation is a weak probe here.** One neuron moves about 1% of next-token probability. If other neurons carry the same signal, removing one can hide a real use. Ablating a facet's co-firing neurons together, or patching paths to specific later components, would be stronger.
- **Logit-space signatures are dominated by improbable tokens.** The top tokens of a specific signature are fragments like "EVER" and "osis" that barely matter for prediction, so point 3 is mostly about how the network moves tokens it wasn't going to predict. The probability view (point 2 and the hand readings) fixes that, but it was added after seeing the results and isn't calibrated against a negative control.
- **Probable next tokens cap help.** A quarter of examples already give the next token over 90% probability, so removing any one neuron can barely change it. That pulls facet medians toward 0.
- **Examples cluster by document.** 1528's "Mendeleev" facet is one book, and half of 2976's "Turkish" facet is one table. Clustered examples make p-values too small and reliabilities too high. Resampling by document would fix that.
- **Only the next 5 predictions are measured.** A neuron's write at one position can be read much later.
- **Bigger facets drift a little.** 2976's Turkish facet picks up Hindi and code, and 1059's price facet picks up street names. With a 0.5 cutoff instead of 0.4, 13 more neurons drop to too few examples (16 in all) and the other 24 stay "no use found".
- **The controls are limited.** The positive control covers only near-identical contexts. The negative control patches the neuron where it doesn't fire, while facet pairs use its natural activations, and it splits other neurons' facets once (even and odd examples) for their reliability, instead of 50 random splits.
- **Two harmful facets out of 101 is no more than chance gives** at p < 0.05. 1528's p is far smaller, but it comes from one book.
- **Scope is unchanged:** one layer of one model, 40 neurons selected with the sentence model, final checkpoint only.

### v3 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v3.py`](recgen/rubric_v3.py) | Pre-registered facet enlargement, measures and labels |
| [`results/facets_v3.json`](results/facets_v3.json) | The enlarged facets: window, position, activation, similarity, whether v1 had it |
| [`results/classification_v3.json`](results/classification_v3.json) | v1, v2 and v3 labels; per-facet help, reliability, top tokens; per-pair scores; sensitivity; controls |
| [`results/v3_neurons.md`](results/v3_neurons.md) | Per neuron: a summary table, then each facet's help, top tokens and examples, and each pair's score |
| [`results/figures/`](results/figures/) `v3_*.png` | Shared-job scores against the controls, help on each least-related pair, label counts |

```bash
./run_v3.sh
```

`run_v3.sh` needs v1's `results/` and `data/` and v2's `results/classification_v2.json`. From scratch it takes about an hour on an M1 Pro: it downloads the final checkpoint once, deletes it after use, and caches 0.9 GB in `data/`. With that cache, it reruns in 30 seconds and gives identical results.

## v4 results

v2 judged adjacency at the final checkpoint, which can't see two facets that grew out of shared circuitry and separated afterwards. v4 measures v2's driver overlaps at all 14 checkpoints after initialization and asks whether two facets shared drivers *while they formed*. It also splits attention into its 192 heads, because v2's MLP-neuron drivers covered only about a quarter of a facet's rise. The rules are in [`recgen/rubric_v4.py`](recgen/rubric_v4.py), committed (`7b4f2ce`) before any checkpoint was recorded. No v1, v2 or v3 file or result changed.

### v4 method

1. **Recording.** At each checkpoint from step 128 to 143k, the same facet examples and background text as v2, run through layer 12. With the LayerNorm scale frozen at its value on each example (as in v2), each pre-activation splits exactly into direct terms: upstream MLP neurons, the 192 attention heads, the token embedding, and biases. The terms add back up within 5e-5 at every checkpoint, and at the final checkpoint v4's MLP driver overlaps match v2's exactly.
2. **Formation window.** A facet's gain is the neuron's median pre-activation on its examples minus its median on background text. The facet forms from the checkpoint after which its gain stays above 25% of its final value to the one after which it stays above 75%. A pair's window spans both facets' windows; for the least-related pairs it runs a median of step 512 to 13k.
3. **Adjacency at formation.** A pair is adjacent at formation if its largest driver overlap inside its window beats the 95th percentile of the same statistic for null pairs (each of the two facets against every other traced neuron's facets, read through this neuron's weights, over the same window). Taking the window's maximum on both sides keeps the false-positive rate near 5% however long the window is. MLP-neuron drivers (top 100, as in v2) are the primary measure; head drivers (top 5 of 192) are reported beside them.
4. **Branched then separated:** adjacent at formation but not at the end (v2). That's the recursive-generalization signature v2 couldn't see.

### What v4 found

| Label (all 40) | v1 | v2 | v4, MLP drivers | v4, head drivers | Expected by chance (v4, MLP) |
|---|---|---|---|---|---|
| recursive generalization | 0 | 2 | 1 | 1 | 0.9 |
| one origin, timing unresolved | 0 | 0 | 0 | 0 | 0.1 |
| packing-like | 35 | 32 | 33 | 32 | 30.3 |
| mixed from the start | 5 | 6 | 6 | 7 | 8.6 |

The 23 hand-picked neurons are 19 packing-like and 4 mixed from the start by MLP drivers, as in v2 (18 and 5 by head drivers).

![Driver overlap of each hand-picked neuron's least-related facet pair over training](results/figures/v4_formation_overlap.png)

Below, "unrelated" means not adjacent at the end by v2's driver test, and "related" means adjacent.

1. **No unrelated pair branched then separated.** By MLP drivers, 0 of 68 unrelated pairs were adjacent at formation, where the null's false-positive rate predicts 3.1. By head drivers it's 0 against 1.4. The least-related pair's overlap peaks at a median 0.05 inside its window, against a median threshold of 0.14. For the hand-picked neurons it stays mostly between 0.00 and 0.05 all through training, and their largest single-checkpoint values are 0.12: 2918 at step 256 (below its threshold of 0.15) and 3656 at step 128 (before its window opens). Across all 40 neurons, the largest are 4010 (0.18) and 2976 (0.13); see point 4.
2. **Zero is below chance, which says more about the pairs than the method** (exploratory, as are points 4 to 6). 0 against 3.1 has a Poisson probability of about 0.05, and an uncorrected check per checkpoint finds 13 of 475 in-window tests (2.7%) above that checkpoint's null 95th percentile, against 5% nominal. These pairs were chosen as each neuron's least alike, while the null includes same-genre facets from other neurons, so they share even fewer drivers than the null. The 68 pairs also aren't independent: they share facets and null sets.
3. **The test catches sharing that persists; brief sharing is less certain.** 17 of 18 related pairs are adjacent at formation by MLP drivers (head drivers: 9 of 18). But these pairs were picked for sharing drivers at the end, so this control covers sharing that lasts, not the brief shared phase v4 is looking for; only the synthetic unit test covers that. Real overlaps of 0.11 to 0.14 sit near the thresholds, and a brief shared phase of about 0.1 in a window that opens before step 1k would face thresholds of 0.12 to 0.29.
4. **Two neurons share drivers between their least-related facets, but steadily, not by branching.** Neuron 4010 (scientific vs legal citations) is labeled recursive generalization by both measures, as in v2: its facets overlap 0.08 to 0.18 at every checkpoint, above that checkpoint's null 95th percentile at 10 of 14. That's the most consistent sharing in the set, and its two facets may be one job (citations). Neuron 2976 (mid-word pieces of rare words) drops from v2's recursive generalization to packing-like, but only because noisy early checkpoints set its window's threshold at 0.143. Its overlap reaches 0.11 by step 8k (where the window closes, against that checkpoint's null 95th percentile of 0.05) and 0.13 at the end, above the per-checkpoint null at 7 of 14 checkpoints. Its head drivers overlap completely at step 2k. Both look like persistent sharing, which is the essay's "one job" picture more than either branching or packing.
5. **v4's formation timing doesn't back v1's emergence order.** The labels still use v1's emergence order, as pre-registered. But for 38 of 40 least-related pairs the two formation windows overlap, and of the 33 pairs v1 calls staggered, the v1-later facet starts forming first in 12 and at the same checkpoint in 8. So the split between "packing-like" and "mixed from the start" mostly reflects v1's 0.5 cutoff: most facet pairs formed over the same stretch of training.
6. **Head drivers judged by heads at the end too.** The pre-registered "branched then separated" compares head adjacency at formation with v2's MLP adjacency at the end. Judging the end by heads as well (the pair's final overlap above its null's final 95th percentile), 3 pairs are head-adjacent at formation but not at the end: 3656 0-1, 2976 0-1 and 2442 0-2. All three share MLP drivers at the end, so none is an unrelated pair, and the count in point 1 stands.
7. **A facet's support moves from MLP neurons to attention during training** (exploratory). Pooling every facet's rise over background, upstream MLP neurons carry 84% at step 128 and 67% at 1k. Attention heads overtake them between 1k and 2k, and by the end heads carry 80% and MLP neurons 20%. (Median per-facet shares put the crossover between 2k and 3k; early per-facet shares are noisy because 22 of 101 facets have no positive rise yet at step 128.) The token embedding stays near zero. These are direct terms only: a head's term can include attention to the current token, and attention patterns weren't analyzed.

![Median share of a facet's rise by source, over training](results/figures/v4_sources.png)

### What this does and doesn't show

- **v4 found no sign that unrelated facets shared upstream drivers while they formed, but the test has limited power.** MLP drivers carry a minority of the rise over most of the formation windows, head drivers are a blunt probe, and the positive control covers lasting sharing, not brief sharing (point 3). So this is absence of evidence for branching, not evidence that facets were attached separately. v1's timing doesn't add much either (point 5), and v3's output test couldn't decide.
- **The clearest sharing is steady, not branching.** 4010 and 2976 share drivers between their least-related facets through most of training (point 4). Neither shows the branch-then-separate pattern.
- **It doesn't show why unrelated facets share a neuron.** v4 doesn't test the packing mechanism (unrelated things placed together because they rarely collide), and there are other possibilities; for example, v1 point 4's repurposed neurons keep an old job as a leftover facet.
- **It doesn't contradict the essay's main claim.** The essay is about most neurons, and these are the 40 most mixed-looking of 4,096 in one layer of a small model. The essay itself says smaller models "have broader neurons", so it expects more mixing here. What this tests is its explanation for the mixed ones, and in this sample it found no evidence for that explanation.

### What's still shaky

- **The rules weren't written blind.** They were committed before any v4 data, but chosen knowing v2's final-checkpoint overlaps and v2's dense pre-activation gain curves. The readings of 4010 and 2976 (point 4) are post hoc.
- **Top-100 overlap is coarse.** Sharing below a facet's top 100 MLP drivers, or through indirect paths (an early neuron acting through later ones), is invisible.
- **Head drivers are a weak probe,** and they carry most of the rise after step 1k to 2k. Top 5 of 192 heads catches only 9 of 18 related pairs, so shared circuitry that lives in attention could be missed. Comparing what the heads attend to and which features they move would be finer.
- **Early training is hard to read.** At steps 128 to 512 the null's 95th percentile reaches 0.28 (step 256), and window thresholds reach 0.29, so brief sharing very early would need a large overlap to show.
- **The grid is log-spaced.** Formation windows span several checkpoints, but a short shared phase between two of them could be missed. Pythia has a checkpoint every 1,000 steps.
- **Facets are defined by their final examples.** A branch whose early form fired on different text than its final examples wouldn't show up in these overlaps.
- **Scope is unchanged:** one layer of one model, 40 neurons selected with the sentence model.

### v4 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v4.py`](recgen/rubric_v4.py) | Pre-registered formation windows and adjacency at formation |
| [`recgen/formation.py`](recgen/formation.py) | Attribution at each checkpoint, with attention split by head |
| [`results/classification_v4.json`](results/classification_v4.json) | v1, v2 and v4 labels; per pair: windows, overlap series, null thresholds, branched-then-separated; source shares by step |
| [`results/v4_neurons.md`](results/v4_neurons.md) | Per neuron: formation windows and each pair's test |
| [`results/figures/`](results/figures/) `v4_*.png` | Least-related pair overlap over training, label counts, source shares |

```bash
./run_v4.sh
```

`run_v4.sh` needs v1's and v2's `results/` and `data/corpus.npy`. From scratch it takes about an hour on an M1 Pro: it downloads 14 checkpoints one at a time, deletes each after use, and caches 3.0 GB in `data/v4/`. With that cache, it reruns in about 5 minutes.

## v5 results

v1 to v4 studied 40 neurons in one layer of one small model. The essay expects bigger models to have narrower neurons, so fewer mixed-looking ones. v5 asks whether that holds: it runs v1's facet detector, unchanged, on six Pythia sizes trained on the same data in the same order. The rules are in [`recgen/rubric_v5.py`](recgen/rubric_v5.py), committed (`58fefef`) before any model was recorded. No earlier result changed (scipy is now a declared dependency).

### v5 method

1. **Models and layers.** Pythia 70M, 160M, 410M, 1B, 1.4B and 2.8B, all in float16. 6.9B doesn't fit in this laptop's 16 GB. At each size, three layers: a quarter, half and three quarters of the way through (layer `int(depth * layers)`). The middle layer is primary.
2. **Neurons.** In each layer, near-dead neurons (peak below half the layer's median peak) and dense ones (v1's rule) are dropped, and 2,048 of the rest are sampled with a fixed seed, all of them in 70M.
3. **Measure.** v1's facet detector on the same 2.1M-token corpus: each neuron's 50 strongest windows, embedded with the sentence model and clustered. The primary measure is the share of sampled neurons that are **unrelated mixed**: 2 or more facets whose least-related pair has centroid similarity below 0.1. The secondary measure is the share with 2 or more facets of any kind, v1's selection criterion. Each share gets a 95% bootstrap interval over neurons.
4. **Verdict** on the primary measure at the middle layer: "fits the essay" if the share falls with size (Spearman correlation with size rank at most -0.8) and 2.8B's interval lies entirely below 70M's; "against the essay" for the mirror image; otherwise "no clear trend". The other depths and the secondary measure get the same test, reported beside it.

As a check, the 410M middle layer is v1's layer 12: v5 finds 2+ facets in 25.3% of its neurons, against v1's 24.9% with float32 and v1's eligibility rule.

### What v5 found

![Share of mixed-looking neurons by model size, at three depths](results/figures/v5_scale.png)

| Unrelated mixed | 70M | 160M | 410M | 1B | 1.4B | 2.8B | Spearman | Verdict |
|---|---|---|---|---|---|---|---|---|
| **middle (primary)** | **16.0%** | **10.6%** | **9.9%** | **8.2%** | **8.1%** | **6.9%** | **-1.00** | **fits the essay** |
| quarter | 13.1% | 11.4% | 11.0% | 10.1% | 11.2% | 10.4% | -0.71 | no clear trend |
| three quarters | 13.6% | 14.5% | 16.0% | 16.1% | 21.0% | 18.7% | +0.94 | against the essay |

Exact intervals and the secondary measure are in [`results/v5_models.md`](results/v5_models.md) and [`results/classification_v5.json`](results/classification_v5.json). Points 2 to 7 are exploratory, added after an independent review of the first results.

1. **At the middle layer, mixed-looking neurons get rarer with size, as the essay predicts.** The unrelated-mixed share falls from 16.0% (14.0 to 17.7%) in 70M to 6.9% (5.8 to 8.1%) in 2.8B, and the share with 2+ facets of any kind falls from 41.0% to 21.5% (also "fits the essay"). More than half of the drop is between 70M and 160M, and 1B to 1.4B is flat (8.2% vs 8.1%). Without 70M the primary measure still fits: 160M's interval (9.4 to 12.0%) sits above 2.8B's.
2. **Part of that decline is the detector finding fewer groups at all.** At the middle layer, neurons whose top 50 contexts form no facet rise from 20.7% to 36.8%. A neuron's strongest contexts are also less alike in bigger models (median pairwise similarity 0.076 to 0.055), and the share of them that land in any facet halves (27% to 14%). So to the sentence model, bigger models' neurons look more varied, not narrower. Conditioning shows where the trend lives. Among neurons with at least one facet, the unrelated-mixed share still falls (20.1% to 10.9%, "fits the essay", with or without 70M). Among neurons with 2+ facets, the share whose facets are unrelated barely moves (38.9% to 32.0%, no clear trend).
3. **After the first step, detector settings matter.** Re-clustering the middle layer at a cosine distance of 0.7, or a minimum facet size of 6, still fits, with or without 70M. A distance of 0.8 fits over all six sizes but not without 70M. A minimum facet size of 4 gives no clear trend (Spearman -0.77).
4. **Other depths don't follow the middle.** A quarter of the way in, the share drifts down only slightly (13.1% to 10.4%). Three quarters of the way in, it rises (13.6% to 18.7%, peaking at 21.0% at 1.4B), which the pre-registered test calls "against the essay". It rises under every other detector setting too (Spearman +0.37 to +0.89), though not all of them pass the full test.
5. **Shared next tokens don't explain the deep-layer rise, but this check is crude.** At three-quarters depth, an unrelated-mixed neuron's two least-related facets share their most common next token 0.5 to 2% of the time, against 0.2 to 0.8% for facets of different neurons (averaged over 200 shuffles). That's above chance but rare, and no higher than at the middle layer (0.9 to 5.7%). The check only catches identical tokens, so a neuron that promotes a class of tokens, like the article neuron in point 6, slips through. A real test would read the neuron's output weights through the unembedding.
6. **Some "unrelated mixed" neurons look like one pattern that the detector splits** (from reading the three random examples per model and depth in [`results/v5_models.md`](results/v5_models.md)).
   - 2.8B middle neuron 3791 fires on spelling pieces: "rag|azzi" and "rig|uarda" in Italian, "Rag|lan" in an English place name, "flag|stone" in a description.
   - 2.8B deep neuron 4166 fires on articles: "the" in English legal and news text, "un" and "des" in French fiction.
   - Others look genuinely mixed, such as 2.8B middle neuron 9176 ("stepped" in fiction vs numbers in math problems) and 1.4B deep neuron 6657 (legal text vs "b|end" in TikZ diagrams).
   - v5 has no checkpoints, so these are relations between facets, not evidence of how they grew.
7. **The dense-neuron filter treats 70M differently.** It drops 24 to 25% of 70M's neurons at the middle and three-quarters layers, against 0.3 to 7% in every other layer. Point 1's result without 70M doesn't depend on it.

### What this does and doesn't show

- **The middle-layer result fits the essay, but it's thinner than the headline.** Mixed-looking neurons get rarer with size, most clearly from 70M to 160M. Beyond that the decline is small and depends on detector settings (point 3). Part of it is the detector finding no groups at all in more of the bigger models' neurons, whose strongest contexts look more varied rather than narrower. Among neurons that do split into groups, the unrelated share barely changes (point 2).
- **The trend depends on depth.** Deep layers go the other way, and nothing here explains why. The detector's blind spots may differ by depth: if deep neurons in bigger models track features that cut across genres and languages, like the article neuron, the detector would count more of them as mixed. That's a guess, not something v5 tested.
- **It measures how mixed neurons look, not why.** v5 counts neurons whose strongest examples split into unrelated groups. It doesn't say whether they got that way by packing, branching or anything else.

### What's still shaky

- **One detector, which behaves differently with size.** The sentence model splits a concept by genre or language and can't see relations the network uses (point 6), and the share of neurons it finds no groups in grows with size (point 2). So the absolute shares overstate real mixing, and comparisons across sizes aren't free of detector effects either.
- **Six sizes, one family, one corpus.** A Spearman correlation over six points is coarse, and the largest model is 2.8B, far below the trillion-parameter models the essay discusses. The 2.1M-token corpus is the same Pile sample for every size.
- **Relative depth has a size-related offset.** `int(depth * layers)` picks the middle layer, but an MLP sits halfway through its block, so its actual relative position is (layer + 0.5) / layers: 0.58 for 70M's middle layer, 0.54 for 160M's and 0.52 to 0.53 for the rest. Shares rise steeply from the middle to the three-quarters layer, so this could inflate 70M's middle share somewhat. Recording neighboring layers would settle it.
- **Sampling.** 2,048 neurons per layer (70M: all 1,546 to 2,002 eligible). Intervals cover neuron sampling, not the choice of corpus or detector settings.
- **Precision.** v5 runs every size in float16, v1 to v4 ran 410M in float32. The 410M check above agrees to within 0.4 points.

### v5 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v5.py`](recgen/rubric_v5.py) | Pre-registered models, depths, eligibility, measures and verdict |
| [`recgen/sweep.py`](recgen/sweep.py) | Window maxima at three depths of each Pythia size |
| [`recgen/analyze_v5.py`](recgen/analyze_v5.py), [`recgen/plots_v5.py`](recgen/plots_v5.py) | Detector, shares, verdicts and exploratory checks; the figure |
| `tests/test_rubric_v5.py`, `tests/test_sweep.py`, `tests/test_analyze_v5.py` | Tests for the rules, the recorder and the analysis |
| [`results/classification_v5.json`](results/classification_v5.json) | Verdicts, per-layer shares and intervals, facet counts, per-neuron facet counts and similarities, exploratory checks |
| [`results/v5_models.md`](results/v5_models.md) | Per model: shares by depth, and three random unrelated-mixed neurons per depth with examples |
| [`results/figures/v5_scale.png`](results/figures/v5_scale.png) | Shares by model size and depth |

```bash
./run_v5.sh
```

`run_v5.sh` needs only `data/corpus.npy`. From scratch it takes about 2 hours on an M1 Pro: it downloads the six models one at a time (about 12 GB in all), deletes each after use, and caches 2.3 GB in `data/v5/`, including sentence embeddings of 918,218 example contexts. With that cache, it reruns in a few minutes.

## v6 results

Every result so far groups a neuron's strongest examples with a sentence model (MiniLM). v5's review found that this detector behaves differently with model size, splits one pattern by genre or language, and that v5's middle-layer decline beyond 70M depends on its settings. The essay's argument is that a neuron should be read the way the network reads it. v6 reruns v5's sweep, on the same neurons and the same examples, with a detector that groups examples by the network's own representation of them. The rules are in [`recgen/rubric_v6.py`](recgen/rubric_v6.py), committed (`e1b34cc`) before any v6 data was recorded. No earlier file or result changed.

### v6 method

1. **Embedding.** An example is represented by the residual stream entering the neuron's layer at that position, which is what the layer's MLP reads through its LayerNorm. The vector is LayerNorm-normalized, the mean over 1,024 random corpus positions is removed, and the neuron's own input direction (its input weights times the LayerNorm gain) is projected out. Without that last step the examples would look alike by construction, since the neuron's pre-activation is exactly that direction's dot product with the vector.
2. **Calibration.** A cosine similarity means different things in MiniLM's space and in each model's residual stream, so the thresholds come from a null: the similarities between all pairs of the 1,024 random positions, embedded the way the neuron's examples are. The percentiles are where v1's thresholds sit in MiniLM's own null on the same positions. Groups merge while their average similarity is above the 98.2nd percentile (MiniLM's 0.25), and two facets are unrelated if their centroids are less similar than the 81.0th percentile (MiniLM's 0.1). So both detectors merge groups at the same level relative to random text.
3. **Unchanged from v5:** the six sizes, the three depths, the sampled neurons (the recorder checks they are v5's), their 50 strongest windows, the minimum facet size of 5, both measures, the bootstrap and the verdict rule. Recording reuses v5's positions. As a check, each neuron's activations rebuilt from the stored residuals match v5's within 0.035 (median top activation 2 to 3).
4. **Comparison with v5,** pre-registered: per-neuron agreement (Cohen's kappa on the unrelated-mixed flags, and the adjusted Rand index between the two detectors' groupings of each neuron's 50 examples), and whether each of v5's verdicts holds. Verdicts without 70M are secondary.
5. **Detector checks,** reported for both detectors: the share of neurons with no facet, how alike a neuron's top examples are relative to the null, token purity (the share of a facet's examples on its most common activating token), and power. For power, 500 synthetic neurons per layer each pool the 25 strongest examples of two sampled neurons, and the check is the share called unrelated mixed. If a detector's power falls with size, a falling share can't be told apart from lost sensitivity.

Before committing the rules, I tried the recorder and detector on Pythia 14M and 31M, which aren't in the sweep. They showed the detector wasn't degenerate, and that it splits mostly by activating token in those small models, so token purity was added as a check. No threshold or rule changed.

### What v6 found

![Share of mixed-looking neurons by model size under both detectors](results/figures/v6_scale.png)

| Unrelated mixed | 70M | 160M | 410M | 1B | 1.4B | 2.8B | Spearman | v6 verdict | v5's verdict |
|---|---|---|---|---|---|---|---|---|---|
| **middle (primary)** | **44.5%** | **44.9%** | **54.2%** | **45.8%** | **47.2%** | **44.0%** | **-0.03** | **no clear trend** | fits the essay |
| quarter | 50.8% | 49.8% | 49.2% | 43.7% | 42.2% | 40.6% | -1.00 | fits the essay | no clear trend |
| three quarters | 43.7% | 47.4% | 42.2% | 52.8% | 45.1% | 46.6% | +0.26 | no clear trend | against the essay |

Intervals, the secondary measure and per-layer agreement are in [`results/v6_models.md`](results/v6_models.md) and [`results/classification_v6.json`](results/classification_v6.json). Points 2 to 6 include exploratory checks: the anisotropy figure in point 6 was added after seeing the results, and the flag rates and grouping overlap (point 3), the excess power (point 4) and the pair-level cutoff (points 2 and 6) after an independent review.

1. **v5's middle-layer decline doesn't survive the change of detector.** Under the network-native detector, the unrelated-mixed share at the middle layer stays near 45% from 70M (42.0 to 47.0%) to 2.8B (41.9 to 46.1%), peaking at 54.2% in 410M. Without 70M there's still no clear trend (Spearman -0.30), and the share with 2+ facets stays at 82 to 88%. So v5's primary verdict doesn't hold, with or without 70M.
2. **No other depth gives a trend that holds across settings.**
   - At three quarters, v5's rise isn't clear under v6's pre-registered settings (no clear trend). But two of the four other settings give "against the essay" over all six sizes (distance 0.8, Spearman +0.83; minimum facet size 4, +0.93), though neither does without 70M.
   - At a quarter, v6 finds a steady fall from 50.8% to 40.6% ("fits the essay", also without 70M), where MiniLM drifted down without passing the test (13.1% to 10.4%). That fall holds under every other setting and without the projection, but not with the LayerNorm gain applied or with the pair-level cutoff in point 6.
3. **The two detectors disagree about which neurons are mixed, though their groupings overlap.** v6 flags 2 to 6 times as many neurons as v5 does, and Cohen's kappa on the flags is 0.03 to 0.15 per layer. v6 flags 48 to 70% of the neurons v5 calls unrelated mixed, and 40 to 52% of the rest. Their groupings of a neuron's examples agree more: of the example pairs MiniLM puts in one facet, v6 also puts 49 to 83% together, against 14 to 33% of all pairs. The median adjusted Rand index between the two groupings is low (0.01 to 0.13), mostly because MiniLM leaves most examples out of any facet.
4. **MiniLM barely tells a pooled pair of neurons from a single neuron, at any size.** Pooling the 25 strongest examples of two random neurons should make a synthetic neuron look mixed.
   - **MiniLM** calls 9 to 23% of these pools unrelated mixed. That's 0 to 10 points more than single neurons, and at most 6 points beyond 70M; at 410M's middle layer it's 9.8% against 9.9%.
   - The pre-registered check reads MiniLM's power as falling with size at the quarter and middle depths (middle: 22.2% to 9.6%). But the pre-registered measure counts the single-neuron share as power, and that share falls too. So MiniLM isn't losing sensitivity with size; it had little to begin with.
   - **v6** calls 55 to 86% of the pools unrelated mixed, 9 to 41 points more than single neurons. At the middle depth that excess is flat after 70M (41 points at 70M, 22 to 25 after), which supports reading v6's flat middle-layer share as real rather than lost power. At a quarter it stays between 18 and 32 points. At three quarters it falls unevenly, from 40 points at 70M to 9 at 2.8B, so v6 loses power there in bigger models.
5. **v6 finds groups in almost every neuron, at every size.** It finds no facet in 0 to 3% of neurons, where MiniLM finds none in 18.5 to 36.8%, rising with size at the quarter and middle depths. Both detectors see a neuron's top examples as less alike, relative to random text, in bigger models. At the middle layer, the median similarity between top examples falls from the 91st to the 79th percentile of the null under v6, and from the 65th to the 57th under MiniLM. Under MiniLM that showed up as more neurons with no facet. Under v6 it doesn't change the number of facets ([checks](results/figures/v6_checks.png)).
6. **Settings don't produce a middle-layer trend, except one.** At the middle depth there's no clear trend under any of these: the other v5 detector settings mapped through the null (shares from 30% to 66%), the LayerNorm gain applied, no projection, or the unrelated cutoff applied to the mean similarity between two facets' examples instead of their centroids (53 to 64%). Only the uncentered variant falls with size (38% to 11%). A likely reason, not tested directly: the mean that every position shares makes up a growing part of the residual vector. It's 26% of the squared norm in 70M's middle layer and 55% in 2.8B's (73% at 2.8B's three-quarters layer). Without centering, a big model's examples would then look more alike whatever they are, which is why the rubric removes the mean.
7. **What the network-native detector groups by** (from the three random neurons per model and depth in [`results/v6_models.md`](results/v6_models.md)). It finds coherent groups that MiniLM misses, for example:
   - 2.8B middle neuron 3663 has math operators ("128\*v *-* 130\*v") and "imm" in "immortality".
   - 2.8B middle neuron 9147 has word pieces ("linuxbrew.b*int*ray", "Bar*th*olomew") and contrast words ("while", "whereas").

   MiniLM found one facet in 3663 and none in 9147. But v6 also splits what reads as one pattern, by token and syntactic role.
   - 2.8B quarter-depth neuron 3737 gets six facets that read to me as one broad pattern of quantities and measurement: time units, delays and intervals, counts and iterations, degrees and distances, and integer types and a math package in code.
   - 2.8B quarter-depth neuron 6726 gets four facets of argumentative prose, split mostly by the activating word ("credibility *of*", "but *that*", "nothing more *than*", "our valued *concepts*"), where MiniLM saw one genre.

   In small models and early layers its facets are mostly single tokens (median token purity 1.0 at 70M's quarter layer). At 1B and up they mix tokens about as much as MiniLM's do (0.33 to 0.5 against 0.29 to 0.5). Token purity also depends on facet size, which differs between the detectors, so this is a rough comparison.

![Detector checks by model size](results/figures/v6_checks.png)

### What this does and doesn't show

- **v5's scale result depends on the detector.** At v1's merge level, a detector that reads the layer's own representation finds the middle layer's share of mixed-looking neurons flat from 70M to 2.8B, near 45%. That holds under every setting tried except the uncentered one. MiniLM barely tells a pooled pair of neurons from a single one (point 4), so v5's shares say little about pooling. Its decline mostly reflects how often MiniLM finds groups at all (v5 point 2). Under v6, no depth gives a trend that holds across settings (point 2). So the repo no longer has a robust scale result for or against the essay's expectation that bigger models have narrower neurons.
- **It doesn't say which detector is right.** Their groupings overlap well above chance, but they disagree about which neurons are mixed (point 3). In the examples I read, MiniLM tends to group by topic and genre, and the residual stream more by the token and its role in the sentence. "Mixed-looking" depends on who reads the examples, and neither reader is ground truth.
- **It measures how varied a neuron's strongest contexts are, not why.** v6 asks how alike the contexts are in the network's representation apart from the neuron's own input direction. A neuron with one job in varied contexts (the essay's V shape) counts as mixed here too. Like v5, it doesn't say whether neurons got that way by packing, branching or anything else.

### What's still shaky

- **The operating point is matched on random text, not on the neurons' examples.** Matching percentiles of a random-text null puts the two detectors' merge levels on the same footing. But a neuron's top examples sit much higher in v6's null than in MiniLM's (a median 72nd to 91st percentile against 57th to 66th), which is why v6 finds groups in nearly every neuron. The absolute shares (about 45%) depend on these choices. Other settings move the middle-layer share between 30% and 66%, though none gives a trend there.
- **The unrelated cutoff compares centroids against a null of single pairs.** How much averaging raises a centroid's similarity depends on facet size and coherence, which differ between the detectors and with model size. So the matched percentile isn't an exact match for the centroid cutoff. Applying the cutoff to the mean similarity between two facets' examples instead raises v6's shares by 5 to 16 points. That gives no clear trend at any depth, so it removes the quarter-depth fall.
- **One view of the network.** v6 reads the residual stream entering the neuron's layer. It doesn't use what downstream components do with the neuron (v3's output side) or other layers' views. Projecting out one direction removes the neuron's linear read, not everything correlated with it (without the projection, results are similar: point 6).
- **v6 loses power in the deep layers of bigger models** (point 4). At three quarters, its excess over single neurons falls from 40 points at 70M to 9 at 2.8B, so a trend at that depth would be hard to read in either direction.
- **The rules weren't written blind.** They were committed before any v6 data, but chosen knowing v5's results and its review, and after a pilot on 14M and 31M. Point 6's anisotropy figure, the post-review checks and the hand readings are post hoc.
- **Scope is unchanged:** six sizes of one family up to 2.8B, one 2.1M-token corpus, float16, 2,048 sampled neurons per layer.

### v6 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v6.py`](recgen/rubric_v6.py) | Pre-registered embedding, calibration, comparison with v5 and detector checks |
| [`recgen/sweep_v6.py`](recgen/sweep_v6.py) | Normalized residuals at v5's example positions and at random positions, and the neurons' input weights |
| [`recgen/analyze_v6.py`](recgen/analyze_v6.py), [`recgen/plots_v6.py`](recgen/plots_v6.py) | Both detectors on the same neurons, verdicts, agreement, checks and sensitivity; the figures |
| `tests/test_rubric_v6.py`, `tests/test_sweep_v6.py`, `tests/test_analyze_v6.py` | Tests for the rules, the recorder and the analysis |
| [`results/classification_v6.json`](results/classification_v6.json) | Verdicts beside v5's, per-layer shares, intervals, agreement, checks and sensitivity, per-neuron results under both detectors |
| [`results/v6_models.md`](results/v6_models.md) | Per model: both detectors' shares and agreement by depth, and three random v6 unrelated-mixed neurons per depth with examples |
| [`results/figures/`](results/figures/) `v6_*.png` | Shares by size under both detectors, and the detector checks |

```bash
./run_v6.sh
```

`run_v6.sh` needs `data/corpus.npy`, v5's `data/v5/` cache (window maxima and sentence embeddings) and `results/classification_v5.json`. From scratch it takes about 2 hours on an M1 Pro: it downloads the six models one at a time (about 12 GB in all), deletes each after use, and caches 4.9 GB in `data/v6/`. With that cache, it reruns in about 15 minutes.

## v7 results

v5 and v6 found that how mixed a neuron looks depends on who reads its strongest examples. The essay's own numbers ("about 90% of neurons have one main job") come from a third kind of reader: language models reading the examples. v7 puts three readers on the same units, including synthetic units with known answers, at the middle layer of all six sizes:

- **MiniLM:** v5's sentence-model detector.
- **network:** v6's detector, using the model's own representation.
- **Haiku:** `claude-haiku-5-5` through the `claude` CLI, with no tools and a fixed system prompt. It sees a unit's 50 examples in a seeded random order, with the firing token marked and no activation values, and answers A (one job), B (related jobs) or C (unrelated jobs). C counts as "unrelated mixed".

The rules are in [`recgen/rubric_v7.py`](recgen/rubric_v7.py), committed (`c17b9eb`) before any language-model call on repo data. Three format-check readings came next, with the prompt unchanged, and they count toward the run. No earlier file or result changed.

### v7 method

1. **Units, per model (300):**
   - **Real:** 100 of v5/v6's sampled neurons.
   - **Pooled:** 150 pools of two random neurons' strongest examples, 50 at each mix (50/50, 70/30 and 90/10). These have a known answer: they mix two neurons.
   - **Related:** 50 pools of two neurons whose input directions are the most alike (picked greedily, no neuron twice). These are a check on false alarms.
2. **Pre-registered tests, per reader:**
   - **Power:** 50/50 pools flagged at least 20 points more than real neurons, with separated 95% intervals, in at least 5 of 6 models.
   - **Specificity:** related pools flagged at least 20 points less than 50/50 pools, in at least 5 of 6 models.
   - **Agreement:** Cohen's kappa between readers on real neurons.
   - **Scale:** v5's verdict rule on each reader's real-neuron flag rate.
3. **Retest:** 60 units (10 per model) were read by Haiku a second time, to measure its consistency.

All 1,860 readings returned a valid label: 6.8M input tokens, which the CLI reported as $2.08 of quota. Every answer, with Haiku's stated patterns, is in [`results/v7_llm_answers.json`](results/v7_llm_answers.json).

### What v7 found

![Three readers on the same units](results/figures/v7_readers.png)

| Reader | Power (models of 6) | 50/50 pools over real, per model | Over all models (95% interval) | Real flagged, 70M → 2.8B | Scale verdict |
|---|---|---|---|---|---|
| MiniLM | **0** | -9 to +18 points | +3 (-1 to +8) | 17% → 3% | fits the essay (Spearman -1.00) |
| network | **4** | +6 to +40 | +22 (+16 to +28) | 42% → 44% (peak 56%) | no clear trend (+0.12) |
| Haiku | **2** | +11 to +32 | +21 (+14 to +27) | 38% → 46% | no clear trend (+0.68) |

Points 1 to 3 and 5 include exploratory checks (the over-all-models effect, the power simulation, the kappa ceilings, shared windows, the retest interval), added after an independent review of the first results.

1. **No reader passes the pre-registered power test, but the test was nearly impossible to pass.** With 50 pools and 100 real neurons per model, a reader whose 50/50 pools are truly 21 points above real neurons passes the test in a given model 42% of the time, and in 5 of 6 models 5% of the time (simulated with the test itself). The fair comparison is the effect over all models:
   - The network reader and Haiku detect pooling about equally: +22 and +21 points. v6's larger check of the network reader (500 pools per layer) gave +22 to +41 at the middle layer of every size, so its misses at 160M (+11) and 2.8B (+6) here are mostly small samples.
   - MiniLM barely detects it: +3 points (-1 to +8). It separates pools from real neurons only at 1B (+18, with separated intervals), and never by the 20-point margin.
2. **The related control was easier than intended, so specificity says little.** All three readers flag related pools less than random 50/50 pools: Haiku in all six models (22 to 48 points less), the network reader in five, MiniLM in one. But related pools often contain two examples from the same text window (38% of them, against 17% of random 50/50 pools and none of the real units, which take one example per window). Haiku flags related pools with a shared window 17% of the time, against 32% without. So the related control is easier than an average real neuron, and Haiku's 6 of 6 mostly reflects that. Judged against real neurons instead, it would pass in 2 of 6 and the network reader in 4.
3. **MiniLM's flags are a conservative subset, not a disagreement.** On real neurons, kappa is 0.20 to 0.38 between Haiku and the network reader, and lower with MiniLM (-0.01 to 0.22 with Haiku, 0.03 to 0.17 with the network reader). But MiniLM flags so few neurons that its kappa with Haiku can be at most 0.07 to 0.53, depending on the model, and at 410M and 2.8B it reaches that ceiling: every real neuron MiniLM flags, Haiku flags too. Over all models, Haiku flags 61% of the neurons MiniLM flags and 41% of the rest, and the network reader 66% and 46%. So MiniLM picks out a small, more mixed subset.
4. **MiniLM's scale decline isn't corroborated.** On these 100 real neurons per model, MiniLM's flag rate falls from 17% to 3%, in the same direction as v5's full sample (16.0% to 6.9%). The network reader is flat, and Haiku drifts up (38% to 46%). Neither passes the trend test. They also have only about 21 points of power and 100 neurons per model, so a decline of MiniLM's size would be near the limit of what they could show.
5. **Haiku is a noisy reader.** Read a second time, 60% of units get the same label and 68% the same flag (kappa 0.37, interval 0.12 to 0.60, from 60 units of which 20 are real neurons). Its agreement with the network reader (kappa 0.20 to 0.38) is about as high as its agreement with itself, so that agreement may be near Haiku's ceiling. Label noise is part of the intervals above; mainly, it pulls kappas down.
6. **Haiku calls 42 to 57% of real neurons one job and 33 to 47% unrelated jobs.** That's far from the essay's "about 90% one job", but the setups differ. Haiku is a small model reading 50 snippets once, without activations, tools or experiments. The essay's agents investigate each neuron and work on larger models. Its stated patterns ([`results/v7_models.md`](results/v7_models.md)) are specific and often checkable:
   - On 70M, "a relative pronoun right after a head noun" and "a word fragment the next token completes (*were*|wolf, *is*|omer, *can*|oe)".
   - Its "unrelated" calls can be reasonable. For 2.8B MLP neuron 560, it lists HTML markup, plain prose tokens like "declined" and articles, word fragments, non-Latin text, and numeric and URL fragments, and most of the 50 snippets are indeed ordinary prose.
   - It rarely uses "related jobs" (8 to 12% of real neurons).

### What this does and doesn't show

- **The scale result has no support beyond MiniLM.** MiniLM barely detects two neurons pooled together, and the two readers that partly do (+21 and +22 points) don't show its decline. That doesn't prove MiniLM's decline wrong: its flags pick out a small subset that the other readers also tend to flag, and they could barely detect a decline of that size. But "bigger models have fewer mixed-looking neurons" now rests on the one reader that can't see known mixing.
- **It strengthens the methods finding.** Cleanliness numbers from clustering top examples with a sentence model shouldn't be trusted without a positive control like the pooled neurons. And a reader's power, measured on known mixtures, matters more than its flag rate.
- **It doesn't test the essay's method.** The essay's agents are larger models that run experiments on each neuron. v7's Haiku reads a fixed set of snippets once. What it shows is that a simple language-model reader is noisy (point 5) and detects pooling about as well as the network reader, which is only moderately well (point 1). An investigating agent may do much better. Running the same positive control on that kind of agent is the obvious next test.
- **No reader here is a gold standard.** None separates pools from real neurons by more than about 22 points on average, and they disagree on many individual neurons. "What share of neurons are mixed?" has no reader-independent answer in this repo.

### What's still shaky

- **Small samples per model.** 50 pools and 100 real neurons per model make per-model tests weak (point 1). The over-all-models effects are the more reliable numbers.
- **The related control is easier than intended** (point 2): pools can contain two examples from the same window, while real units can't. Pooled units also use only each neuron's strongest 25 to 45 examples, while real units include ranks 26 to 50.
- **The readers see different inputs.** MiniLM embeds the context before the token, Haiku sees about 30 tokens around it (24 before, 4 after), and the network reader sees the whole preceding window. Haiku also sees no activation values, which makes its task harder than a real audit.
- **The CLI adds about 500 tokens of its own to each call.** Its length is the same for every kind of unit, so it can't tell Haiku which units are pooled, but its content isn't documented.
- **One small model, one prompt, one reading per unit** (plus 60 retests). Another model or prompt could change the rates. The prompt was fixed before any reading, and none was tuned.
- **Pooled units mix two neurons' strongest examples.** Real polysemantic neurons may mix less cleanly, with weaker second jobs or shared contexts, so the pools test the easiest kind of mixing to detect.
- **Scope:** the middle layer only, Pythia 70M to 2.8B, 300 units per model.

### v7 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v7.py`](recgen/rubric_v7.py) | Pre-registered units, prompt, labels and tests |
| [`recgen/units_v7.py`](recgen/units_v7.py), [`recgen/llm_reader.py`](recgen/llm_reader.py) | Real, pooled and related units; the cached language-model reader |
| [`recgen/analyze_v7.py`](recgen/analyze_v7.py), [`recgen/plots_v7.py`](recgen/plots_v7.py) | Three readers' flags, rates, power, specificity, agreement, retest; the figure |
| [`results/v7_llm_answers.json`](results/v7_llm_answers.json) | Every Haiku reading: label, stated patterns, tokens, and the prompt's SHA-256 so its integrity can be checked |
| [`results/classification_v7.json`](results/classification_v7.json) | Per-model rates with intervals, verdicts, kappas, retest, per-unit flags |
| [`results/v7_models.md`](results/v7_models.md) | Per model: rates by reader and unit type, and units where the readers disagree, with Haiku's patterns |
| `tests/test_rubric_v7.py`, `tests/test_llm_reader.py`, `tests/test_analyze_v7.py` | Tests for the rules, the reader and the analysis |

```bash
./run_v7.sh
```

`run_v7.sh` needs v6's `data/v6/` recordings, v5's MiniLM embeddings, and the `claude` CLI signed in. Haiku's answers are cached in `data/v7/` (not committed) and collected in `results/v7_llm_answers.json`. With the cache, it makes no model calls and reruns in about a minute. From scratch, the 1,860 readings take about 45 minutes, six at a time. Their labels will differ somewhat, because Haiku isn't deterministic (point 5).

## v9 results

v7's positive control pooled two neurons' strongest examples by fixed counts (25 + 25, 35 + 15, 45 + 5). That is the easiest mixing to detect: both halves are at full strength, and pools can repeat a text window, which real neurons can't. v9 builds a more natural synthetic two-job neuron. The **union** of two real neurons fires, in each window, at the larger of the two neurons' raw activations, and its 50 strongest examples come from its own ranking, one per window. How many examples each neuron contributes is then set by their activations, as it would be for a real neuron with two jobs. The rules are in [`recgen/rubric_v9.py`](recgen/rubric_v9.py), committed (`bace9af`) before any reading and amended once (`76bd70d`), also before any reading (see below). No earlier file or result changed.

### v9 method

1. **Units.** 150 unions of two random sampled neurons per model, at the middle layer of the six sizes. Because a union takes a maximum, its top 50 windows always come from the two neurons' own top 50s, so every example was already embedded for MiniLM (v5) and recorded for the network reader (v6). Only Haiku needed new readings. Real neurons are v7's 100 per model, with v7's Haiku readings reused (same units, same prompt).
2. **Amendment.** The first rules scaled each neuron by its top-50 level. Building the units showed that this makes the two neurons cross at about their 25th example, so 829 of 900 unions came out exactly 25/25, which defeats the point. Raw activations give minority shares from 0 to 25 of 50 (a median of 2 to 7 by model), so raw became primary. The two scalings are reported as sensitivity checks for MiniLM and the network reader. This was decided before any reader ran.
3. **Pre-registered tests, per reader:**
   - detection: union minus real flag rate, averaged over models, with a bootstrap interval;
   - the same for v7's 50/50 pools minus unions;
   - dose-response: flag rate by how many examples come from the minority neuron (0 to 4, 5 to 14, 15 to 25), top bin minus bottom bin;
   - a Haiku retest of 10 unions per model.

900 union readings and 60 retests all returned a valid label.

### What v9 found

![Natural unions by minority share](results/figures/v9_unions.png)

| Reader | Unions minus real | Flag rate by minority share: 0-4 / 5-14 / 15-25 | Top minus bottom | v7 pools minus unions |
|---|---|---|---|---|
| MiniLM | -1 (-4 to +2) | 9% / 9% / 9% | **0** (-5 to +4) | +4 (0 to +8) |
| network | -2 (-7 to +3) | 31% / 56% / 73% | **+42** (+35 to +50) | +24 (+18 to +30) |
| Haiku | -4 (-9 to +1) | 22% / 52% / 64% | **+43** (+35 to +50) | +25 (+19 to +31) |

Points 1, 2 and 6 and the scaling checks in point 3 are pre-registered; the rest is exploratory. The comparison of balanced unions with real neurons (point 3), point 4 and the partial correlations in point 5 were added after an independent review of the first write-up; their numbers are in `exploratory_after_review` in the results.

1. **The network reader and Haiku respond strongly to how mixed a union is; MiniLM doesn't respond at all.** Both readers' flag rates climb by more than 40 points from unions dominated by one neuron to balanced ones. MiniLM stays near 9% whatever the mix.
2. **No reader flags natural unions more than random real neurons overall** (the pre-registered detection test). Most raw unions are dominated by one neuron: 484 of 900 take fewer than 5 of their 50 examples from the minority neuron, and 279 take none, so they are exactly one real neuron's top 50.
3. **Balanced unions are detected about as well as v7's pools.** Unions with 15 to 25 minority examples are flagged more than real neurons by 24 points (17 to 32) by the network reader and 22 (14 to 29) by Haiku, close to v7's 50/50 pool margins (+22 and +21). MiniLM: -1 (-5 to +4). So the gap between natural unions and v7's pools in the table comes from dominated unions, not from natural mixing being harder to see. The scaling sensitivity checks agree for the network reader: with each neuron scaled to its top-50 level (829 of 900 unions come out exactly 25/25), it flags unions 23 points (18 to 28) more than real neurons, and 16 (11 to 21) with peak scaling.
4. **Part of the 42-point climb is which neuron a union is made of, not mixing.** A union dominated by one neuron is mostly that neuron's own top examples, and raw activations pick the neuron with the larger activations, which look cleaner (point 5). Comparing each union with its own two neurons' flags, weighted by how many examples each contributes:

   | Minority share | Network reader: unions flagged | Their own neurons, weighted | Excess |
   |---|---|---|---|
   | 0-4 | 31% | 32% | -1 |
   | 5-14 | 56% | 52% | +5 |
   | 15-25 | 73% | 47% | **+26** |

   So mixing adds about 26 points for balanced unions, and dominated unions look like their own neurons. MiniLM's excess is within 2 points of 0 in every bin. Haiku has readings of only 100 real neurons per model, too few of the unions' source neurons for the same check; point 3 is its closest comparison.
5. **Neurons with larger peak activations look cleaner, and most of that is how alike their strongest examples are.** By quartile of a neuron's peak activation (its largest window maximum), from weakest to strongest:

   | Reader | Flagged unrelated mixed, by peak quartile | Spearman by model |
   |---|---|---|
   | network (all sampled neurons: 2,048 per model, 1,561 in 70M; v6's flags) | 46-66% / 50-61% / 41-56% / **24-34%** | -0.15 to -0.30 |
   | Haiku (v7's 100 real neurons per model) | pooled 62% / 51% / 40% / **19%** | -0.25 to -0.39 |
   | MiniLM (v5's flags) | 7-19% / 9-17% / 7-18% / 5-9% | 0.00 to -0.09 |

   Strong neurons' 50 strongest examples are more alike in the network's own representation (v6's top-pair percentile, the median similarity of the 50 examples ranked against random positions). Holding that coherence fixed (partial Spearman), the link between peak and the flag disappears: from -0.15 to -0.30 to between 0.00 and +0.21 for the network reader, and from -0.32 to -0.06 for Haiku, pooled over 600 neurons. For the network reader this is partly circular, since its flag is computed in that representation; Haiku never sees it. Token purity (the share of the 50 examples on the most common activating token) explains almost none of it: holding it fixed leaves -0.15 to -0.29, and -0.31 for Haiku. This was found after looking at the data, so v12 tests it on held-out depths and fresh neurons.
6. **Haiku may be more consistent on unions than on v7's units:** a second reading of 60 unions gives the same label 80% of the time and the same flag 85% (kappa 0.69, against 0.37 on v7's 60 retest units). The units differ, and 60 retests give wide intervals, so this is only suggestive.

### What this does and doesn't show

- **This MiniLM detector can't see mixing.** With the mix varied naturally from 0 to 25 examples, its flag rate doesn't move (point 1), and balanced unions look like real neurons to it (point 3). Together with v7, that's strong evidence that clustering top-50 examples with this sentence model and v1's settings doesn't measure polysemanticity. Other sentence models or settings weren't tested.
- **The network reader and Haiku can, in proportion to how mixed a unit is,** once the union's own neurons are accounted for (points 3 and 4). But they flag about 45% of real neurons, and flag a neuron less the more alike its strongest examples are (point 5). So their absolute rates mix polysemanticity with how coherent a neuron's top examples are.
- **The magnitude effect is real in these data but mostly a coherence effect.** Strong neurons have more coherent top examples, and coherence is what both readers respond to. Whether weak neurons are more often doing two jobs, or just have more diffuse top examples, v9 can't say.

### What's still shaky

- **Unions aren't real polysemantic neurons.** A union takes the larger of two neurons' activations. A real neuron with two jobs computes them together, and its jobs may share contexts or differ in strength in ways a union doesn't.
- **Raw activations select the stronger neuron,** so the pre-registered detection test (point 2) compares unions dominated by strong neurons with random real neurons. Points 3 and 4 correct for that, but they're exploratory.
- **25 drawn pairs were replaced** (10 in 70M, 0 in 2.8B) because tied activations put a union example outside both neurons' recorded top 50s.
- **The magnitude analysis reuses data** that earlier versions already looked at, and uses only the middle layer.
- **Haiku runs through the `claude` CLI,** which adds a little hidden context to each call: the working directory, the platform, the date and an account notice. It includes no git branch, commit or file contents, so it doesn't reveal which units are unions.
- **Scope:** the middle layer, Pythia 70M to 2.8B, 150 unions per model.

### v9 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v9.py`](recgen/rubric_v9.py) | Pre-registered unions, scaling (with the amendment), bins and tests |
| [`recgen/units_v9.py`](recgen/units_v9.py), [`recgen/read_v9.py`](recgen/read_v9.py) | Union units; Haiku readings through v7's cached reader |
| [`recgen/analyze_v9.py`](recgen/analyze_v9.py), [`recgen/plots_v9.py`](recgen/plots_v9.py) | Three readers' flags, the tests, sensitivity, the magnitude check; the figure |
| [`results/classification_v9.json`](results/classification_v9.json), [`results/v9_llm_answers.json`](results/v9_llm_answers.json), [`results/v9_models.md`](results/v9_models.md) | Results, every Haiku reading with its prompt hash, and per-model examples |

```bash
./run_v9.sh
```

`run_v9.sh` needs v5's and v6's caches and v7's results. With v9's cached Haiku answers in `data/v9/`, it makes no model calls and reruns in a few minutes. From scratch, the 960 readings take about 30 minutes.

## v12 results

v9 found, after looking at the middle layer, that neurons with a larger peak activation are flagged unrelated mixed less often. v12 is a confirmatory test of that finding on data it didn't use. The rules are in [`recgen/rubric_v12.py`](recgen/rubric_v12.py), committed (`49e15c4`) before any held-out data was examined. They were amended once (`d102a67`), after the first analysis run crashed and before any result was computed. No earlier file or result changed.

An independent review of v9 ran while v12's rules were being written. Its report, which included magnitude numbers at the other depths, was read after the rules were committed. Its main finding, that strong neurons' top 50 examples are more alike, became v12's exploratory point 5.

### v12 method

1. **Held-out data.**
   - The network reader's (v6) and MiniLM's (v5) flags at the quarter and three-quarters depths of all six sizes. They were computed for every sampled neuron in earlier versions but never examined for magnitude.
   - Fresh Haiku readings at the middle layer: 25 sampled neurons from each peak quartile per model (600 in all), none of them among v7's 100, read with v7's prompt and CLI. All 600 returned a valid label.
   - v11's Haiku readings of real neurons (v7's design at the other depths, branch `v11`), added if they existed at analysis time. The first write-up of v12 wrongly said no v11 readings existed and left them out; v12's independent review caught that. By then the quarter depth was complete (100 real neurons per model, 600 in all), and the three-quarters depth was added when v11 finished reading it (599 of 600 with a valid label).
2. **Peak** is log(peak / median peak of the layer's sampled neurons). The normalization doesn't affect any test, since quartiles and correlations are taken within models and the regression has a fixed effect per model.
3. **H1 (primary), per reader and depth.** The top peak quartile's flag rate minus the other three quartiles' must be below zero, with a bootstrap interval entirely below zero. Peak's Spearman correlation with the flag must also be negative in at least 5 of the 6 models.
4. **H2 (controls).** A logistic regression of the flag on standardized log peak, peakedness (peak over the 50th-ranked window maximum), density (share of windows at or above half the peak) and token purity, with a fixed effect per model. It holds if log peak's coefficient has a bootstrap interval entirely below zero.
5. **Amendment.** One neuron (70M, quarter depth) fires in only 16 windows, so its peakedness is undefined. It stays in H1 and is left out of H2.

The middle-layer network and MiniLM results are where the effect was found. They're reported as "discovery", apart from the held-out tests. Haiku's fresh neurons are new neurons, but at the discovery depth.

### What v12 found

![Flag rate by peak quartile](results/figures/v12_magnitude.png)

| Reader, depth | Neurons | Flagged, by peak quartile (weakest to strongest) | Top minus rest (points), models negative | Log-peak coefficient with controls | H1 / H2 |
|---|---|---|---|---|---|
| **Held out** | | | | | |
| network, quarter | 12,242 | 54% / 53% / 48% / 30% | -21 (-23.5 to -19.5), 6/6 | -0.58 (-0.64 to -0.52) | yes / yes |
| network, three quarters | 11,786 | 52% / 53% / 50% / 30% | -21 (-23.4 to -19.4), 6/6 | -0.56 (-0.62 to -0.51) | yes / yes |
| Haiku, quarter (v11's real neurons) | 600 | 50% / 41% / 37% / 17% | -25 (-33.0 to -17.6), 6/6 | -0.85 (-1.19 to -0.59) | yes / yes |
| Haiku, three quarters (v11's real neurons) | 599 | 44% / 41% / 32% / 20% | -20 (-27.7 to -12.0), 6/6 | -0.81 (-1.13 to -0.54) | yes / yes |
| Haiku, middle (fresh neurons) | 600 | 59% / 50% / 44% / 13% | -38 (-44.9 to -30.5), 6/6 | -0.94 (-1.37 to -0.60) | yes / yes |
| MiniLM, quarter | 12,242 | 11% / 11% / 12% / 10% | -1 (-2.4 to +0.1), 4/6 | -0.22 (-0.31 to -0.14) | no / yes |
| MiniLM, three quarters | 11,786 | 17% / 17% / 19% / 13% | -5 (-6.4 to -3.5), 6/6 | -0.28 (-0.35 to -0.21) | yes / yes |
| **Discovery** | | | | | |
| network, middle | 11,801 | 56% / 55% / 49% / 28% | -26 (-27.6 to -23.9), 6/6 | -0.57 (-0.62 to -0.51) | yes / yes |
| MiniLM, middle | 11,801 | 10% / 11% / 11% / 7% | -4 (-5.2 to -3.1), 5/6 | -0.35 (-0.44 to -0.26) | yes / yes |

Points 4 and 5 are exploratory. Point 5 was added after v9's review; point 4, the correlations in point 5 and its paired interval were added after v12's review. Their numbers are under `exploratory_with_coherence` and `exploratory_after_v12_review` in the results.

1. **The effect replicates on every held-out test for the network reader and Haiku, in all six models.**
   - Network reader: at both held-out depths, the strongest quarter of neurons is flagged 21 points less than the rest. It's mostly a step at the top quartile (54%, 53%, 48%, then 30%), with Spearman correlations of only -0.08 to -0.30 by model.
   - Haiku: 25 and 20 points less on about 600 real neurons at each held-out depth, and 38 points less on 600 fresh neurons at the middle depth. Haiku's decline is more graded (59%, 50%, 44%, 13% at the middle depth).
2. **The pre-registered controls don't explain it.** With peakedness, density and token purity in the regression, log peak's coefficient stays clearly negative in every test (H2).
3. **MiniLM shows the same direction at a much smaller scale:** 1 to 5 points, and its quarter-depth H1 fails. Its H2 holds, but its flag rate is only 10% to 19%.
4. **The effect doesn't depend on using the single peak, or on neurons Haiku had already seen.**
   - With quartiles of the mean of a neuron's 50 strongest window maxima (the level of the examples the readers see) instead of its single largest, the top quartile is flagged 23 to 27 points less by the network reader and 22 to 32 less by Haiku, in all six models.
   - 84 of Haiku's 600 fresh neurons are part of v9's Haiku-read unions, so Haiku had in effect seen some of their examples. Without them, H1 is -38 points (-46 to -30) and H2's coefficient is -1.08 (-1.53 to -0.73).
5. **Strong neurons' top examples are more alike inside the network, and that similarity accounts for much of Haiku's effect.** v6's top-pair percentile measures how alike a neuron's 50 strongest examples are in the network's representation: their median pairwise similarity, ranked against random positions. It correlates 0.50 to 0.60 with log peak. Adding it to H2's regression:
   - **Haiku**, which never sees that representation: similarity's coefficient is -0.93 (-1.28 to -0.69) at the middle depth, -0.81 at the quarter depth and -0.85 at three quarters. Log peak's coefficient shrinks by 0.58 (0.42 to 0.77, paired bootstrap) at the middle depth, from -0.94 to -0.37 (-0.81 to -0.03); by 0.52 at the quarter depth; and by 0.60 at three quarters, where its interval then includes 0 (-0.56 to +0.10).
   - **MiniLM:** the same pattern, smaller. Similarity's coefficient is -0.17 to -0.39, and log peak's shrinks by 0.10 to 0.25; at three quarters its interval then includes 0.
   - **Network reader:** similarity takes over (-1.4 to -1.5), and log peak's coefficient turns positive (+0.18 to +0.27). This says little. The flag and the similarity are computed from the same similarity matrix of the same examples, against the same null, so this is close to controlling for the outcome. The positive sign is a suppression effect of that, not evidence that stronger neurons are flagged more at fixed similarity.

### What this does and doesn't show

- **A pre-registered, replicated regularity.**
  - The network reader confirms it at two held-out depths of six Pythia sizes.
  - Haiku confirms it on real neurons at both held-out depths and on fresh neurons at the middle depth.
  - It holds with the mean of the top 50 in place of the single peak.
- **Its source is shared with how alike a neuron's strongest examples are.** That similarity is a continuous version of what the readers judge, so point 5 describes the effect rather than explaining it. Even so, it's notable that Haiku, which reads only text, tracks a similarity measured inside the network.
- **It doesn't say weak neurons are more polysemantic.** Two readings fit the data, and both predict lower similarity for weak neurons:
  - Mixing could really concentrate in weakly firing neurons.
  - Weak neurons could be tuned more broadly, so their strongest examples are more diffuse without being two jobs.
  - The toy model of superposition predicts something related within single neurons: the largest activations are clean and smaller ones more polysemantic ([Elhage et al. 2022, "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html), section "Computation in Superposition", subsection "Superposition vs Sparsity"). Here, though, a weak neuron's top 50 doesn't reach further down its own range. The median 50th example sits at 0.76 to 0.80 of the peak in the weakest quartile and 0.64 to 0.76 in the strongest, by model and depth (`rank_50_fraction_of_peak`).
  - Telling the two readings apart needs examples from a fixed band of each neuron's own range, or a causal test of whether a weak neuron's groups of examples have different effects downstream.
- **It's partly known.** A link between strong, rare, heavy-tailed activation and cleaner units has been reported for SAE features, for GPT-2's universal neurons ([Gurnee et al. 2024](https://arxiv.org/abs/2401.12181), section 4.3) and in toy models. Peak activation predicting how mixed a raw neuron's top examples look, across a model family and confirmed by two readers, appears to be a modest extension.

### What's still shaky

- **The flag is a property of the top-50 sample, not of the neuron.** Any selection that makes strong neurons' top 50 more alike would give this pattern.
- **Point 5 is close to circular for the network reader** (same similarity matrix and null), and post hoc for all readers.
- **Haiku's labels are noisy.** v7's retest gave kappa 0.37 on real neurons, and v12 didn't retest.
- **Token purity** is computed with a stable sort, so in about 10% of neurons one of its 50 windows differs from the readers' 50 at a float16 tie (purity changes by at most 0.02).
- **Haiku runs through the `claude` CLI,** which adds a little hidden context: the working directory, the platform, the date and an account notice. It includes no git branch, commit or file contents.
- **Scope:** Pythia 70M to 2.8B, three depths, the 50 strongest examples per neuron. v5's eligibility filter drops near-dead and very dense neurons, so the weakest quartile isn't the models' weakest neurons.

### v12 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v12.py`](recgen/rubric_v12.py) | Pre-registered held-out data, H1 and H2, with the amendment |
| [`recgen/read_v12.py`](recgen/read_v12.py) | Fresh middle-layer neurons per peak quartile; Haiku readings through v7's cached reader |
| [`recgen/analyze_v12.py`](recgen/analyze_v12.py), [`recgen/plots_v12.py`](recgen/plots_v12.py) | Covariates, both hypotheses per reader and depth (with v11's readings), the exploratory checks; the figure |
| [`results/classification_v12.json`](results/classification_v12.json), [`results/v12_llm_answers.json`](results/v12_llm_answers.json) | Results and every fresh-neuron Haiku reading |

```bash
./run_v12.sh
```

`run_v12.sh` needs v5's and v6's caches and results, and reads v11's cached readings in `data/v11/llm/` if they're there. With v12's cached Haiku answers in `data/v12/`, it makes no model calls and reruns in about four minutes.

## v19 results

Every result so far comes from one model family. Pythia's blocks are GPT-NeoX's, where attention and the MLP read the same residual stream in parallel. v19 asks whether the repo's two methods findings hold in a family with a sequential residual, GPT-2, whose MLP reads the residual stream after its own block's attention:

- **(a)** MiniLM barely tells two pooled neurons from one, while the network reader and Haiku do (v7).
- **(b)** Neurons with a larger peak activation are flagged unrelated mixed less often (v12).

It's a pre-registered replication, not a new method. The rules are in [`recgen/rubric_v19.py`](recgen/rubric_v19.py), committed (`87deb48`) before any GPT-2 activation was recorded or any unit was read. Before that, the only GPT-2 component run was its tokenizer, on 512 corpus windows, to size the corpus problem in method point 2. Two earlier modules changed without changing any output: [`recgen/corpus.py`](recgen/corpus.py) now exposes v1's packed token stream (rebuilding `data/corpus.npy` from it gives the same array), and `analyze_v6.Layer` takes an optional path. No earlier result changed, and no rule was amended.

### v19 method

1. **Models.** GPT-2 small, medium and large (124M, 355M and 774M parameters; XL was left out for disk space), in float16, at the middle layer: 6 of 12, 12 of 24 and 18 of 36.
2. **Corpus.** The same 4,096 Pile windows, re-tokenized. A 512-token Pythia window decodes to fewer than 512 GPT-2 tokens 29% of the time, so each GPT-2 window is the first 512 GPT-2 tokens of the text that starts where the Pythia window starts in v1's packed stream. Document separators carry over as GPT-2's end-of-text token.
3. **Neurons and readers, rules unchanged.** v5's eligibility rule and a sample of 2,048 neurons per model, with window maxima as in v5 (each window's first 32 positions ignored). The three readers are v7's:
   - **MiniLM:** 32 GPT-2 tokens of context up to the firing token, v1's clustering and v5's unrelated cutoff.
   - **network:** v6's detector, reading the residual stream where GPT-2's MLP reads it: entering `ln_2`, after the block's attention. It's LayerNorm-normalized and centered on 1,024 random positions, with the neuron's input direction (its `c_fc` column times `ln_2`'s gain) projected out. Each unit's thresholds are v6's percentiles of its own null of random positions, recomputed in each GPT-2 layer.
   - **Haiku:** v7's prompt, system prompt and CLI, with snippets of 24 GPT-2 tokens before the marked token and 4 after.
4. **Units,** as in v7, per model: 100 real neurons, 50 pools of two random neurons at each mix (50/50, 70/30, 90/10) and 50 related pools. 60 units in all were read twice.
5. **Pre-registered tests.**
   - **(a)** Per reader, the over-all-models excess: the 50/50 pools' flag rate minus real neurons', averaged over the three sizes, with a 95% bootstrap interval (v11's main measure, and v7's exploratory one). (a) replicates if MiniLM's interval includes 0 or stays below +10 points, and the network reader's and Haiku's intervals lie above 0.
   - **(b)** v12's H1 and H2, on the network reader for all 6,144 sampled neurons and on Haiku for the 300 real neurons. H1: the top peak quartile is flagged less than the other three, with an interval below zero, and peak's Spearman correlation with the flag is negative in all three models (v12 asked for 5 of 6). H2: log peak's coefficient stays below zero with peakedness, density and token purity controlled. (b) replicates if H1 holds for the network reader.

All 960 readings returned a valid label, from `claude-haiku-5-5` (the model v7 used): 3.7M input tokens, which the CLI reported as $1.13. Every answer is in [`results/v19_llm_answers.json`](results/v19_llm_answers.json).

### What v19 found

![v19 on GPT-2](results/figures/v19_gpt2.png)

**(a) replicates.**

| Reader | GPT-2: 50/50 pools over real, mean of three sizes (95% interval) | small / medium / large | Pythia, same measure on v7's units (six sizes) | GPT-2 minus Pythia (three sizes against six) |
|---|---|---|---|---|
| MiniLM | **-2** (-7 to +2) | -5 / 0 / -2 | +3 (-1 to +8) | -5.5 (-12 to +1) |
| network | **+22** (+13 to +31) | +35 / +10 / +22 | +22 (+16 to +28) | 0 (-11 to +11) |
| Haiku | **+22** (+14 to +31) | +16 / +28 / +23 | +21 (+14 to +27) | +1.5 (-9 to +12) |

Post hoc, after review: with token position projected out of GPT-2's residual, the network reader's excess is +35 (+26 to +44; point 7).

**(b) replicates for the network reader; Haiku doesn't confirm it.**

| Reader | Neurons | Flagged, by peak quartile (weakest to strongest) | Top minus rest (points), models negative | Log-peak coefficient with controls | H1 / H2 | Pythia, middle layer (v12): top minus rest; coefficient |
|---|---|---|---|---|---|---|
| network | 6,144 | 49% / 51% / 45% / 27% | **-22** (-24.3 to -19.2), 3/3 | -0.34 (-0.41 to -0.28) | **yes** / yes | -26; -0.57 |
| Haiku (real neurons) | 300 | 26% / 18% / 14% / 11% | **-8** (-17.2 to +1.3), 3/3 | -0.40 (-0.89 to +0.01) | no / no | -38 (fresh neurons); -0.94 |
| MiniLM (secondary) | 6,144 | 3% / 5% / 6% / 5% | -0.3 (-1.6 to +0.8), 0/3 | +0.13 (-0.02 to +0.27) | no / no | -4; -0.35 |

Points 1 to 5 are pre-registered (1 and 2 the tests, 3 to 5 secondary). Point 6 is post hoc: it was noticed after seeing the results. Point 7 and the sentences that start "Added after review" are post hoc too, from an independent review of these results; their numbers are under `post_hoc_after_review` in the results, from [`recgen/position_v19.py`](recgen/position_v19.py) and [`recgen/review_v19.py`](recgen/review_v19.py). GPT-2's numbers are in [`results/classification_v19.json`](results/classification_v19.json), beside Pythia's for the two tests; point 6's Pythia figures come from v6's and v7's results.

1. **(a): MiniLM can't see pooling in GPT-2 either, and the other two readers see it at least as well as in Pythia.** MiniLM's excess is -2 points, with an interval that includes zero, and it's within 5 points of zero at every size. The network reader and Haiku both flag 50/50 pools 22 points more than real neurons, their low ends at +13 and +14. Each reader's GPT-2 excess is within 6 points of its Pythia excess, though those differences have wide intervals (about ±6 to ±11 points). Added after review: those differences set the mean over GPT-2's three sizes (124M to 774M) against the mean over Pythia's six (70M to 2.8B), so they don't compare matched sizes. And token position holds down the network reader's GPT-2 excess: with position projected out it's +35 (+26 to +44), so its 0-point difference from Pythia understates how well it sees pooling in GPT-2 (point 7).
2. **(b): the network reader flags the strongest quarter of neurons 22 points less than the rest,** in all three models (Spearman -0.27, -0.15 and -0.09), close to v12's held-out -21 on Pythia. Like Pythia's, it's mostly a step at the top quartile (49%, 51%, 45%, then 27%). Log peak's coefficient stays negative with the controls (H2), at about 60% of Pythia's (-0.34 against -0.56 to -0.58); one neuron with undefined peakedness is left out of H2, as v12's amendment requires. Haiku points the same way in every model (Spearman -0.21, -0.22 and -0.11) and declines steadily by quartile (26% to 11%), but its interval includes zero, so H1 and H2 don't hold for it. Added after review: with token position projected out, the network reader's top quartile is flagged 24 to 25 points less than the rest, in all three models (point 7).
3. **v7's per-model tests and agreement come out much as on Pythia.** The per-model power test (+20 points with separated intervals) passes for the network reader in 1 of 3 models, Haiku in 2 and MiniLM in none. Related pools are flagged less than 50/50 pools by 20 points or more for the network reader and Haiku in 2 of 3 models each. As in v7 that control is easy: 40 to 46% of related pools contain two examples from one window, against 12 to 22% of 50/50 pools and none of the real units. On real neurons, kappa between the network reader and Haiku is 0.11 to 0.36, and below 0.2 with MiniLM. In GPT-2 small, 0.22 is as high as the two readers' rates allow: the network reader flags all 8 of the real neurons Haiku flags.
4. **Haiku's second readings agree more on GPT-2's units:** a second reading of 60 units gives the same label 87% of the time and the same flag 95% (kappa 0.87, interval 0.71 to 1.00), against 0.37 on v7's Pythia units. Added after review: that more likely reflects which units were retested than a steadier reader. Haiku flagged 27% of them at the first reading, against 45% of v7's retested units, and only 17 of the 60 are real neurons (v7: 20). v11's retest of 120 Pythia units, read earlier the same night, gave kappa 0.53.
5. **As on Pythia, coherence carries Haiku's magnitude trend.** Adding v6's top-pair percentile (how alike a neuron's 50 strongest examples are inside the network) to H2's regression gives it a coefficient of -1.18 (-1.83 to -0.73) for Haiku, and log peak's coefficient rises by 0.80 (0.48 to 1.35), to +0.40 (-0.08 to +1.11). Log peak and coherence correlate at about 0.64. For the network reader this is near circular, as in v12 (coherence -1.51, log peak turning positive). For MiniLM coherence's coefficient is small (-0.14, -0.33 to +0.03), and log peak's turns positive, +0.21 (+0.03 to +0.39).
6. **Post hoc: Haiku calls far fewer GPT-2 neurons unrelated mixed than Pythia neurons.** It flags 8% (3 to 14%), 18% and 25% of real neurons from small to large, against 33 to 47% on Pythia, and calls 87%, 72% and 69% of them one job (Pythia: 42 to 57%). It flags fewer 50/50 pools too (24%, 46%, 48%; Pythia: 56 to 78%), so its excess is unchanged. The network reader's rates on real neurons (40 to 46%) are close to Pythia's. That suggests GPT-2's top examples aren't less varied in the network's own representation: their median top-pair percentile is 90, 83 and 84 by size, inside Pythia's range of 79 to 91 (v6). Both comparisons are made in the representation that token position distorts in GPT-2, though. With position projected out, the network reader flags 24 to 44% of GPT-2's real neurons instead of 40 to 46%, below Pythia's 42 to 56% (v7), and the top-pair percentile wasn't recomputed that way (point 7). So whether GPT-2's top examples are as varied as Pythia's in the network's own terms is open. Why Haiku reads them as cleaner isn't tested here. Haiku's low rate on GPT-2 small also leaves (b)'s Haiku test little room: 8 of its 100 real neurons are flagged.

   Added after review, three explanations the data rule out:
   - **Size.** At the Pythia sizes nearest GPT-2's (160M, 410M and 1B), v7's Haiku flagged 33%, 46% and 47% of real neurons, against GPT-2's 8%, 18% and 25%.
   - **A change in Haiku since v7.** Haiku read Pythia's real neurons earlier the same night (v11's readings ended about 1.5 hours, and v12's about 2.7 hours, before v19's began, by the cache files' times), and flagged 36% and 34% of them at a quarter and three quarters of depth (v11) and 42% of fresh middle-layer neurons (v12).
   - **Repeated contexts, one dominant token or shorter snippets.** In GPT-2's real units, 1.0 to 1.9% of examples share their 9-token context with another example in the unit (Pythia: 0.6 to 2.9%), the most common firing token makes up a median of 16 to 22% of a unit's examples (Pythia: 16 to 31%), and prompts average 6,850 to 7,030 characters (Pythia: 6,530 to 6,800).

   That leaves differences this design can't separate: the corpus (Pythia's training data, off-distribution for GPT-2), the tokenizer and the model family itself.
7. **Post hoc, after review: token position reaches the network reader in GPT-2, and projecting it out strengthens both of the reader's results.** GPT-2 adds a learned absolute position embedding to its residual stream; Pythia's rotary embeddings act inside attention and add nothing to it. So in GPT-2's centered middle-layer residual, two random positions look more alike the closer together they are: over the pairs of the 1,024 random positions, cosine similarity correlates -0.53, -0.56 and -0.52 with the gap between their positions by size, against -0.01 to -0.03 in Pythia's v6 recordings at the middle layer of all six sizes. Since the network reader groups a neuron's examples by similarity, judged against those random positions, position can make examples look alike or unrelated whatever the neuron does. [`recgen/position_v19.py`](recgen/position_v19.py) fixes a position subspace for each layer by one rule: regress every residual stored for the layer (the sampled neurons' 50 strongest examples and the random positions), centered, on ten smooth functions of its position p (as a fraction of 512), namely 1, p, p², log p and the sine and cosine of 2πhp for h = 1, 2 and 3, and take the top 2 or 6 right singular vectors of the fitted values. Position explains 10%, 7% and 5% of the residual's squared norm by size, and the top two directions carry 95 to 99.7% of that. With them projected out, the null's correlation with the position gap is -0.02 to 0.00. Projecting the subspace out of every example and null vector, beside the neurons' own directions, with the pre-registered tests' bootstrap draws:

   | Projected out | Network flags changed | Sampled neurons flagged: small / medium / large | (a) network excess | (b) top minus rest, models negative |
   |---|---|---|---|---|
   | nothing (pre-registered) | | 40.5% / 46.0% / 41.7% | +22 (+13 to +31) | -22 (-24.3 to -19.2), 3/3 |
   | position, 2 dims | 16.7% | 31.3% / 40.4% / 36.6% | **+35** (+26 to +44) | **-25** (-26.9 to -22.4), 3/3 |
   | position, 6 dims | 17.3% | 30.9% / 40.0% / 37.9% | **+36** (+27 to +45) | **-24** (-26.8 to -22.0), 3/3 |
   | 6 random directions (control) | 1.5% | 40.4% / 45.8% / 41.7% | +25 (+15 to +34) | -22 (-24.1 to -18.9), 3/3 |

   Removing position lowers the flag rate on real neurons (by 2 to 17 points by model) and raises it on 50/50 pools in medium and large, so the gap between them widens. Neither verdict changes, and the top quartile is flagged less in all three models (Spearman -0.31, -0.23 and -0.15 with 2 dimensions removed). Six random directions change 1.5% of flags. So position makes some GPT-2 neurons look mixed to the network reader, which hides part of its power on GPT-2; it doesn't create either result.

**Recording checks.**

- Activations rebuilt from the stored residuals match the recorded ones within 0.0038, against median peaks of 3.0 to 3.4.
- 2,896, 3,960 and 5,039 neurons are eligible (of 3,072, 4,096 and 5,120), and 2,048 are sampled in each model.
- MiniLM's null on GPT-2's contexts puts v1's merge level at the 98.3rd percentile and the unrelated cutoff at the 81.2nd, against the frozen 98.2 and 81.0.
- The residual's mean makes up 41%, 64% and 37% of its squared norm by size (Pythia's middle layers: 26 to 55%), which the centering removes.
- **The pre-registered float32 check failed for medium, and the check itself was at fault.** A float32 pass on the CPU matched the float16 window maxima within 0.013 for small and 0.009 for large, but only within 3.97 for medium. A post hoc check ([`recgen/precision_v19.py`](recgen/precision_v19.py), [`results/v19_precision_check.json`](results/v19_precision_check.json)) found that this Mac's CPU float32 path computes GPT-2 medium wrongly: its next-token loss on 4 windows is infinite, against 2.408 in float64 on the CPU, and it's already 11% off at the first block's output, which points to a fault in that path rather than to float32's precision. Float32 and float16 on MPS, the recording's device, match the float64 reference (loss 2.408 and 2.409; hidden states within 2e-6 and 1.3e-3). Those hidden-state differences are relative to each hidden state's largest absolute value, and GPT-2's residual has outsized dimensions, so they understate the error in a typical dimension. Added after review: in medium's middle layer, dimension 268 holds the normalized residual's largest value at 99.8% of the random positions, a median of 20 against 0.37 for a typical coordinate (small and large: dimensions 447 and 440, at 86% and 90%). Float32 on MPS matches the stored float16 window maxima within 0.027 over 64 windows. So the recording stands.
- Added after review: all 960 readings' prompt hashes match the prompts the committed code builds for their units.

### What this does and doesn't show

- **(a) holds across the change of architecture.** In GPT-2, as in Pythia, clustering top examples with this sentence model doesn't separate two pooled neurons from one, while the network's own representation and a language model reading the snippets do, by at least about 22 points in both families (the network reader's GPT-2 excess is +35 with token position projected out, point 7). The warning from v7 and v9, that MiniLM-based cleanliness numbers don't measure mixing, isn't specific to Pythia.
- **(b) holds for the network reader, which is the less independent test.** It gets stronger with token position projected out (point 7). The network reader's flag and the coherence that carries the effect come from the same representation (point 5). In Pythia, Haiku, which never sees that representation, confirmed the effect at three depths. In GPT-2 it points the same way in every model but doesn't pass: it flags few real neurons (point 6) and reads only 300. So in GPT-2 the magnitude effect is shown clearly by one reader and only suggested by the independent one.
- **Absolute "share mixed" numbers don't transfer between these setups.** Haiku's real-neuron rate drops from about 43% on Pythia to 17% on GPT-2 while its excess on pools stays the same. Model size, a change in Haiku, repeated contexts, a dominant token and snippet length don't explain the drop (point 6), but the corpus, the tokenizer and the size ranges change along with the family, so it isn't shown to be a property of GPT-2. A cleanliness figure from one reader in one setup says little about another.

### What's still shaky

- **The corpus is the Pile, Pythia's training data, not GPT-2's.** GPT-2 was trained on WebText, so code, LaTeX and other Pile text may be unusual inputs for it. That could change which examples top a neuron's ranking in ways that don't affect Pythia.
- **Re-tokenization changes what each reader sees.** GPT-2 windows end at different text than Pythia's, and 24 GPT-2 tokens of snippet cover different text than 24 Pythia tokens.
- **The network reader reads a different place in the block.** In GPT-2 the MLP's input includes its own block's attention output; in Pythia it doesn't. That's the right analog, but it means the two families' network readers aren't identical.
- **GPT-2's learned absolute position embedding reaches the network reader.** Positions close together in a window look alike in GPT-2's centered residual, and the pre-registered reader doesn't remove that. Projecting a fitted position subspace out changes about 17% of its flags and moves both of its results (point 7), so its GPT-2 flag rates, and comparisons of them with Pythia's, carry this confound. The subspace's rule was chosen after the review, and other rules could remove somewhat more or less. Projecting position out would also remove a real position-related job, if a neuron had one.
- **Haiku's (b) test is weak:** 300 neurons with a 17% flag rate, against v12's 600 per test with flag rates of 34 to 42%.
- **Three sizes, 124M to 774M, one layer each.** Per-model numbers rest on 50 pools and 100 real neurons per model.
- **Token purity** is computed with a stable sort, so in about 10% of neurons one of its 50 windows differs from the readers' 50 at a float16 tie (purity changes by at most 0.02), as in v12.
- **Why the CPU float32 path fails on GPT-2 medium isn't known** (PyTorch 2.14.1 on this M1 Pro). It doesn't affect the recording, which ran on MPS, but the pre-registered check didn't do its job for that model.
- **Haiku runs through the `claude` CLI,** which adds a little hidden context: the working directory, the platform, the date and an account notice. It includes no git branch, commit or file contents.

### v19 files and reproduce

| Path | What it is |
|---|---|
| [`recgen/rubric_v19.py`](recgen/rubric_v19.py) | Pre-registered models, corpus, readers, units and both tests |
| [`recgen/gpt2_v19.py`](recgen/gpt2_v19.py) | GPT-2's corpus and recording: window maxima, residuals entering `ln_2`, input weights |
| [`recgen/units_v19.py`](recgen/units_v19.py), [`recgen/read_v19.py`](recgen/read_v19.py) | v7's units on GPT-2; Haiku readings through v7's reader, with a $40 spend guard |
| [`recgen/analyze_v19.py`](recgen/analyze_v19.py), [`recgen/plots_v19.py`](recgen/plots_v19.py) | Three readers, tests (a) and (b), secondary results and checks; the figure |
| [`recgen/precision_v19.py`](recgen/precision_v19.py) | Post hoc device and precision check of GPT-2 medium |
| [`recgen/position_v19.py`](recgen/position_v19.py), [`recgen/review_v19.py`](recgen/review_v19.py) | Post hoc, after review: the network reader with token position projected out (point 7); the checks on Haiku's rates, its retests and the recording |
| [`results/classification_v19.json`](results/classification_v19.json), [`results/v19_llm_answers.json`](results/v19_llm_answers.json), [`results/v19_precision_check.json`](results/v19_precision_check.json) | Results with per-unit and per-neuron flags, and the post hoc checks under `post_hoc_after_review`; every Haiku reading with its prompt hash and cost; the precision check |
| `tests/test_rubric_v19.py`, `tests/test_gpt2_v19.py`, `tests/test_read_v19.py`, `tests/test_analyze_v19.py`, `tests/test_position_v19.py`, `tests/test_review_v19.py` | Tests for the rules, the GPT-2 recorder, the spend guard, the analysis and the post hoc checks |

```bash
./run_v19.sh
```

`run_v19.sh` needs `data/corpus.npy`, the `pile-10k` dataset (once, to rebuild v1's stream for GPT-2's corpus), `results/classification_v7.json` and `classification_v12.json`, and the `claude` CLI signed in. The post hoc checks also need v6's middle-layer recordings in `data/v6/` and `results/v7_llm_answers.json` and `v12_llm_answers.json`, and read v11's cached readings in `data/v11/` if they're there. From scratch it takes about 1.5 hours on an M1 Pro: recording takes 46 minutes, downloading the three models one at a time (5.3 GB in all) and deleting each after use; MiniLM embeddings take 5 minutes; the 960 readings take about 26 minutes of reading, six at a time. In this run they didn't come from one `read_v19` call. An ad hoc driver read GPT-2 small's 320 from 02:56 to 03:05, right after small was recorded and before medium and large were (03:12 and 03:38), and `read_v19` read the other 640 from 03:52 to 04:09, reusing small's cache: 72 minutes from the first reading to the last, by the cache files' times. That doesn't change what was read, since every reading's prompt hash matches the prompt the committed code builds. It caches 970 MB in `data/v19/`. With that cache, it makes no model calls and reruns in about 6 minutes. The precision check isn't part of it: `uv run python -m recgen.precision_v19 gpt2-medium` downloads medium again.

## Files

| Path | What it is |
|---|---|
| [`results/candidates.md`](results/candidates.md) | The 40 traced neurons and their facets at the final checkpoint |
| [`results/hand_picked.json`](results/hand_picked.json) | The 23 hand-picked neurons, with why each looks mixed |
| [`results/classification.json`](results/classification.json) | Labels, emergence steps, facet similarities, exploratory stats |
| [`results/trajectories.md`](results/trajectories.md) | Each hand-picked neuron's strongest examples at every checkpoint |
| [`results/figures/`](results/figures/) | Facet response curves (hand-picked and all traced) and label counts |

## Reproduce

```bash
uv sync
uv run pytest
./run.sh
```

`run.sh` takes about 2 hours on an M1 Pro. It downloads 15 checkpoints of about 1.6 GB each, one at a time, and deletes each after use. Recorded activations (about 2.5 GB) go to `data/`, which isn't committed.
