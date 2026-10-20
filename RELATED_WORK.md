# Related work and novelty audit

This file checks which of the repo's findings are new, against published work up to 2026-10-19. It covers papers, preprints and research posts from interpretability labs (arXiv, Transformer Circuits, Distill, OpenAI, EleutherAI, Goodfire, LessWrong and the Alignment Forum, and NeurIPS, ICML, ICLR, ACL and EACL papers). Each claim about a source was checked against the source's own text, and an independent fact-check by a second model went over the citations (see the end of the file).

"Not found" below means not found after the searches listed for that finding. It is not proof that nobody has done it. Short keys like [Bills23] point to the [references](#references).

## What's actually new

| Finding | Verdict | Closest prior work | What the repo adds |
|---|---|---|---|
| 1. How mixed a neuron looks, and the scale trend, depend on the reader | **Partly known** | Separate papers already disagree on how neuron cleanliness changes with size, using different measures and models [Bills23, Elhage22, Gurnee23, Dravid26]. Network-native facet clustering exists in vision and GPT-2 [Nguyen16, Foote24, Dreyer24]. | Not found: three kinds of reader on the *same* neurons and examples, across six sizes of one family, under one verdict rule, where the sign of the scale trend changes with the reader. |
| 2. Pooled-neuron positive control | **Partly known** | Known-answer polysemantic units exist: synthetic "A or B" functions and neurons for interpretability agents [Schwettmann23, Shaham24], a toy benchmark [GuptaKumar26], and sanity checks for explanation metrics [Oikarinen25]. | Not found: a planted mixture built from two *real* neurons' own top examples in a language model, used to measure each reader's power. None of the language-model polysemanticity measures checked here reports power on planted mixtures [GuptaKumar25, Kopf25, Foote24, Liu26]. |
| 3. Checkpoint-level test of recursive generalization | **Not found** | The idea is older than the essay ("conjoined" neurons [Goh21]). Pythia checkpoints have been used to track SAE features and circuits [Xu24, Bayazit26, Quirke23, Tigges24], not the formation order of one neuron's facets. | Apparently the first public test of the essay's mechanism on language-model checkpoints. The result is a null with low power, so it is new as a test, not as a refutation. |
| 4a. Input support moves from MLP neurons to attention heads early (steps 1k to 3k) | **Partly known** | The window matches known early transitions in Pythia: induction heads at about 2B tokens (about step 1k) [Tigges24], the in-context-learning phase change at 2.5 to 5B tokens [Olsson22], a phase transition coinciding with learning-rate warm-up [Quirke23], which ends at step 1,430. | The measurement (direct-term share of a mid-layer neuron's rise, MLP neurons against heads) wasn't found. It likely reflects those known transitions, so it is a modest observation. |
| 4b. Residual-stream mean direction grows with size | **Already published** | Machina & Mercer measured the same quantity (average cosine similarity) for every layer of Pythia 70M to 6.9B: low in small models' middle layers, high in 1.4B's and 6.9B's [MachinaMercer24, Fig. 1]. | Nothing new beyond confirming it on 2.8B and at the repo's layers. Cite it. |
| 4c. Neurons repurposed during training, with an old job left as a facet | **Partly known** | Goh et al. guessed that conjoined features might be "a vestigial phenomenon from early in training" [Goh21]. SAE features in Pythia-410M "alter their primary semantic association during training" [Xu24]. The essay's own InceptionV1 neuron mixed5a:178 changes job over training. | A few neuron-level examples in a language model, from reading checkpoints by hand. Anecdotal. |

### For a post

Supported, as far as this search goes:
- No one seems to have compared different readers head to head on the same neurons across a model family. Here the scale trend in mixed-looking neurons depends on the reader.
- A sentence-embedding reader barely detects two real neurons pooled together (+3 points), while the network reader and an LLM reader detect it moderately (+22 and +21). No published polysemanticity measure checked here reports this kind of power test.
- This seems to be the first public checkpoint test of the essay's mechanism on a language model, and it found no evidence for it in Pythia-410M's most mixed-looking neurons, with limited power.

Not supported:
- "First to show neuron interpretability metrics are unreliable." Many papers show metric problems [Oikarinen25, Bolukbasi21, Bills23, Paulo24].
- "First positive control for polysemanticity." Synthetic polysemantic units with known answers exist [Schwettmann23, Shaham24, GuptaKumar26]. What's new is the pooled-real-neuron design and the reader comparison.
- "Recursive generalization is wrong." The repo's tests are underpowered and cover 40 neurons in one layer of one small model.
- The anisotropy result as a finding of the repo. It is published [MachinaMercer24].

## 1. Reader-dependence of polysemanticity measurements

**The repo's finding** (README, v5 to v7). On the same sampled neurons and the same top-50 examples, at the middle layer of Pythia 70M to 2.8B:
- A sentence-model reader (MiniLM clustering) finds fewer mixed-looking neurons in bigger models (16.0% to 6.9%, "fits the essay").
- A network reader (clustering in the layer's own residual stream, centered, with the neuron's input direction projected out, thresholds calibrated on a random-text null) finds no clear trend (about 45%).
- Per neuron, their flags barely agree (Cohen's kappa 0.03 to 0.15), though their groupings of examples overlap well above chance, and MiniLM's flags are mostly a conservative subset of the others'.
- An LLM reader (Claude Haiku labeling the 50 examples) agrees with the network reader at kappa 0.20 to 0.38, about as high as its agreement with itself on a retest (0.37).

**Closest prior work**

- *Scale trends in neuron cleanliness already depend on the measure, across papers.* These use different models and different notions of "clean", so they can't be compared directly with each other or with the repo. Together they show that the direction of the trend has never been measure-independent.
  - LLM simulation scoring: explanation scores fall with subject-model size, over GPT-3-series models from 98K to 6.7B parameters. The authors add that "these trends may be artificial", reflecting limits of the explanation method [Bills23, section "Subject model size"].
  - Blinded human ratings: about 35% of baseline neurons look interpretable in models of 1 to 40 layers, and 33% at 64 layers, with no strong trend by size [Elhage22, section 6.2, Figure 4].
  - Sparse probing on Pythia 70M to 6.9B: sparsity "increases on average with scale", but some features gain dedicated neurons, some stay the same, and some split across neurons [Gurnee23, section 5.3, Figure 4].
  - Vocabulary-space selectivity (kurtosis of a neuron's output weights against the unembedding) in Pythia, GPT-2, OPT and Qwen-2.5: neurons that recur across models ("Rosetta neurons") get more selective with scale, while the rest don't [Dravid26, section 5.1, Figure 6(a)]. An average over all neurons can hide such diverging subpopulations.
  - The essay's agent audits: "bigger models learn narrower neurons" [Cammarata26].
- *Readers disagree within a study.*
  - PSI's open-vocabulary component, recomputed with a second CLIP text head, correlates at Spearman 0.21 with the first [GuptaKumar25, section 5.4, Figure 6].
  - Five autointerp scoring methods correlate with each other at Spearman 0.15 to 0.79. The sentence-embedding method agrees least with human ratings (0.32) [Paulo24, Table A1].
  - Clustering a vision neuron's top examples by upstream attributions separates concepts better than clustering by activations, judged by CLIP and DINOv2 embeddings [Dreyer24, sections 3 and 4.2].
  - Shared word forms ("bank") inflate standard superposition and polysemanticity metrics, in models from 110M to 70B parameters [AitHouHwa26]. This bears on the repo's observation (v6 point 7) that the network reader splits examples by token and syntactic role.
- *Network-native facet readers.* The repo's network reader is a language-model version of an established idea.
  - Nguyen et al. cluster a unit's top images in the unit's own layer code (or fc7 for output units), with PCA, t-SNE and k-means, to find its facets [Nguyen16, section 2.3, Algorithm 1, section 3.2].
  - Foote's "neuron embeddings" for GPT-2 small take the elementwise product of the residual stream entering the MLP and the neuron's input weights at the top-activating token, cluster them hierarchically, and propose cluster statistics as proxies for polysemanticity [Foote24, sections 3.1 to 3.3]. The repo's v6 does nearly the opposite: it projects the neuron's input direction *out* to see the context apart from what the neuron reads.
- *Null calibration.* PSI turns each component into a z-score against a null (random rotations of the embeddings, permuted labels) [GuptaKumar25, section 3.6]. v6's random-text null plays the same role.

**Verdict: partly known.** Measure-dependence of "how clean are neurons" is not new. What wasn't found is a controlled comparison: different kinds of reader on identical units and examples, across a model family, under one pre-registered verdict rule, showing that the sign of the scale trend changes with the reader. No paper found reports per-neuron agreement between a sentence-embedding reader, a network-geometry reader and an LLM reader on language-model neurons.

**Strengthens or contradicts**
- Strengthens: the prior LLM-reader trend [Bills23] runs against "bigger is cleaner", as the repo's Haiku drift (38% to 46% unrelated) does, and the human-rater result [Elhage22] is flat, like the network reader. Different models and measures, so this is suggestive only.
- Complicates: on Pythia itself, probing finds sparser representations on average at larger scale [Gurnee23], which is in the essay's direction. The repo doesn't measure that, and a reader of top examples and a sparse probe answer different questions.
- Complicates: the essay's readers are investigating agents working on much larger models [Cammarata26]. The repo's readers are far weaker, so its flat or rising rates don't contradict the essay's ~90% figure directly (README v7 says this too).

**Searched:** see the [search log](#search-log), queries 1 to 5 and 10. I also checked the related-work sections of [GuptaKumar25], [Kopf25], [Foote24] and [Liu26].

## 2. The pooled-neuron positive control

**The repo's finding** (README v6 point 4, v7 point 1). A synthetic unit pools the top examples of two random real neurons, so its right answer is "mixed". Over all six sizes, the share of 50/50 pools called unrelated-mixed, minus the share of real neurons, is +3 points for MiniLM (interval -1 to +8), +22 for the network reader and +21 for Haiku. v6's larger check (500 pools per layer) gives MiniLM 0 to 10 points and the network reader 22 to 25 points at the middle depth after 70M.

**Closest prior work**
- *Known-answer polysemantic units for automated interpreters.*
  - FIND includes 60 composed entity functions that respond to either of two concepts, such as "time and duration OR winter sports", scored by Vicuna-13B, to test interpretability agents [Schwettmann23, section 2.3, Table 1]. It reports that complex functions are harder to describe than atomic ones, but no split by polysemantic versus single-concept units.
  - MAIA tests on 85 synthetic vision neurons, including "polysemantic" ones selective for "either A or B", built from Grounded DINO and SAM [Shaham24, section 4.2].
  - DPMI scores neuron polysemanticity against a toy benchmark with known answers (Spearman 0.755, AUROC 0.877, beating seven baselines) and a modular-arithmetic transformer [GuptaKumar26].
- *Sanity checks for evaluation metrics.* The "missing labels" and "extra labels" tests ask whether a metric penalizes a too-specific or too-generic explanation. Accuracy, recall, AUC and top-and-random simulation scoring fail at least one [Oikarinen25, sections 4.2 and 4.4, Table 3]. They treat polysemanticity as an OR of concepts (section 6.1), but don't test merged neurons.
- *Polysemanticity measures without a planted-mixture test.* PSI is validated against random-data nulls and patch interventions [GuptaKumar25, sections 5.1 and 5.3]. PRISM's sanity checks test whether descriptions are faithful [Kopf25, appendix A.6]. Neuron embeddings [Foote24] and NeuronScope [Liu26, appendix D] are checked qualitatively. EleutherAI's autointerp uses shuffled explanations and random SAEs as baselines [Paulo24, Tables A3 and A8] and proposes, but doesn't run, adversarial tests against similar features (appendix D.5).

**Verdict: partly known.** Positive controls with known polysemantic answers exist, mostly synthetic functions for interpretability agents and vision models. Not found: (a) a mixture built from two real neurons' own top examples, so the planted unit keeps real example statistics, and (b) using it to measure a polysemanticity *detector's* power, as excess over real neurons, across readers and model sizes. The specific result, that a sentence-embedding clustering reader barely detects such mixtures, wasn't found elsewhere.

**Strengthens or contradicts**
- Strengthens: sentence-embedding scoring agrees least with human judgments among autointerp scoring methods (Spearman 0.32) [Paulo24, Table A1]. That's a different task (scoring an explanation), but it points the same way as MiniLM's weak power.
- Limits the claim: pooled units are the easiest kind of mixture, as the README says. Known-answer benchmarks [Schwettmann23, Shaham24] also include conditional units (one concept only in the presence of another). The repo doesn't test those, and they are closer to how real mixed neurons might look.

**Searched:** queries 2, 3, 6 and 7 in the [search log](#search-log), and the evaluation sections of [GuptaKumar25], [Kopf25], [Foote24], [Liu26] and [Paulo24].
