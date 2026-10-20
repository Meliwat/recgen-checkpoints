# Related work and novelty audit

This file checks which of the repo's findings are new, against published work up to 2026-10-19. It covers papers, preprints and research posts from interpretability labs (arXiv, Transformer Circuits, Distill, OpenAI, EleutherAI, Goodfire, LessWrong and the Alignment Forum, and NeurIPS, ICML, ICLR, ACL and EACL papers). Each claim about a source was checked against the source's own text, and an independent fact-check by a second model went over the citations (see the end of the file).

"Not found" below means not found after the searches listed for that finding. It is not proof that nobody has done it. Short keys like [Bills23] point to the [references](#references).

## What's actually new

| Finding | Verdict | Closest prior work | What the repo adds |
|---|---|---|---|
| 1. How mixed a neuron looks, and the scale trend, depend on the reader | **Partly known** | Separate papers already disagree on how neuron cleanliness changes with size, using different measures and models [Bills23, Elhage22, Gurnee23, Dravid26]. Network-native facet clustering exists in vision and GPT-2 [Nguyen16, Foote24, Dreyer24]. | Not found: three kinds of reader on the *same* neurons and examples, across six sizes of one family, under one verdict rule, where one reader's falling trend isn't reproduced by the other two. |
| 2. Pooled-neuron positive control | **Partly known** | Known-answer polysemantic units exist: synthetic "A or B" functions and neurons for interpretability agents [Schwettmann23, Shaham24], a toy benchmark [GuptaKumar26], and sanity checks for explanation metrics [Oikarinen25]. | Not found: a planted mixture built from two *real* neurons' own top examples in a language model, used to measure each reader's power. None of the language-model polysemanticity measures checked here reports power on planted mixtures [GuptaKumar25, Kopf25, Foote24, Liu26]. |
| 3. Checkpoint-level test of recursive generalization | **Not found** | The idea is older than the essay ("conjoined" neurons [Goh21]). Pythia checkpoints have been used to track SAE features and circuits [Xu24, Bayazit26, Quirke23, Tigges24], not the formation order of one neuron's facets. | Apparently the first public test of the essay's mechanism on language-model checkpoints. The result is a null with low power, so it is new as a test, not as a refutation. |
| 4a. Input support moves from MLP neurons to attention heads early (steps 1k to 3k) | **Partly known** | The window matches known early transitions in Pythia: induction heads at about 2B tokens, or step 1k [Tigges24, YinSteinhardt25], the in-context-learning phase change at 2.5 to 5B tokens [Olsson22], a phase transition coinciding with learning-rate warm-up [Quirke23], which ends at step 1,430. | The measurement (direct-term share of a mid-layer neuron's rise, MLP neurons against heads) wasn't found. It likely reflects those known transitions, so it is a modest observation. |
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
- Per neuron, their flags barely agree (Cohen's kappa 0.03 to 0.15), though their groupings of examples overlap well above chance. MiniLM flags few neurons, and the other readers flag those more often than the rest (README v7 point 3).
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

**Verdict: partly known.** Measure-dependence of "how clean are neurons" is not new. What wasn't found is a controlled comparison: different kinds of reader on identical units and examples, across a model family, under one pre-registered verdict rule, showing that the verdict on the scale trend changes with the reader. No paper found reports per-neuron agreement between a sentence-embedding reader, a network-geometry reader and an LLM reader on language-model neurons.

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

## 3. A checkpoint-level test of recursive generalization

**The repo's finding** (README v1, v2, v4). Pythia-410M, layer 12, the 40 most mixed-looking neurons, 15 to 29 checkpoints. Pre-registered rubrics label how each neuron's facets arrived.
- v2 labels 2 of 40 neurons recursive generalization by upstream-driver adjacency, against 0.9 expected by chance.
- v4 finds that 0 of 68 unrelated facet pairs shared drivers while they formed, against 3.1 expected from the null.
- The README reads this as no evidence for the mechanism in this sample, from tests with limited power.

**Closest prior work**
- *The mechanism, before the essay.* Multimodal Neurons describes CLIP neurons whose concepts seem "conjoined", "overlapping in a superficial way in one facet, and then generalizing out in multiple directions" (Phil-adelphia / Phil-ippines / Phil-ip). A footnote guesses they may be "a vestigial phenomenon from early in training" [Goh21, section "Feature properties", paragraph "Polysemanticity and Conjoined Neurons"]. The essay credits this paragraph, and says recursive generalization is "easy to check if you have training checkpoints" [Cammarata26].
- *The essay's own evidence.* Vision models it retrained, where it follows neurons over training, for example InceptionV1 mixed5a:178 [Cammarata26, the figure "A feature takes shape over training"]. For language models it reports audits of finished models (Qwen3-32B), not checkpoints.
- *Another origin story.* "Incidental polysemanticity": random initialization can assign several features to one neuron by chance, and training dynamics then strengthen the overlap, even when there are enough neurons [Lecomte24]. That would most likely show up as unrelated facets present from early in training. The repo's rubric has no label for it: such a neuron would be called "mixed from the start", or "packing-like" if one facet crossed the 0.5 threshold later.
- *Pythia checkpoint studies of features and circuits.*
  - Gurnee et al. suggest tracking how neurons "develop and change through time" with Pythia's checkpoints, as future work [Gurnee23, section 6.4].
  - SAE-Track trains a chain of sparse autoencoders over Pythia-410M checkpoints. It reports three phases of feature formation and three patterns: maintaining, shifting and grouping [Xu24, section 3, Figure 2].
  - Crosscoders across three checkpoints of Pythia-1B, OLMo-1B and BLOOM-1B show token detectors giving way to grammatical features [Bayazit26, sections 6.1 and 7.1].
  - In Pythia-70M, a German-detecting neuron's circuit and the n-gram circuits it gates form independently and then join [Quirke23].
  - Circuit components (induction, successor and copy-suppression heads) emerge at similar token counts from 70M to 2.8B [Tigges24, section 3.2].
  - None of these asks whether a single mixed-looking neuron's unrelated facets grew from a shared start.

**Verdict: not found.** I found no published test of recursive generalization on language-model checkpoints. Given that the essay appeared four days before this audit, that isn't surprising. Two cautions for describing it: the result is a null from underpowered tests, so it isn't evidence against the essay. And the scope (40 neurons in one layer of a 410M model) is far from the essay's models.

**Strengthens or contradicts**
- Neither directly. The repo's exploratory observation that most facet pairs formed over the same stretch of training (v4 point 5) fits incidental polysemanticity [Lecomte24] as easily as packing. A rubric that separates the two would need a prediction that distinguishes them.

**Searched:** queries 7 to 11 in the [search log](#search-log).

## 4. Smaller findings

### 4a. Input support moves from MLP neurons to attention heads early in training

**The repo's finding** (README v4 point 7, exploratory). Pooled over facets, upstream MLP neurons carry 84% of a facet's direct-term rise at step 128 and 67% at step 1k. Attention heads overtake them between steps 1k and 2k (median per facet: 2k to 3k) and carry 80% at the end.

**Closest prior work**
- Pythia trains on about 2.1M tokens per step [Tigges24, section 2.3], so steps 1k to 3k are about 2 to 6B tokens. Pythia-410M's config sets `"warmup": 0.01` of `"train-iters": 143000`, so warm-up ends near step 1,430 [Pythia-config].
- Induction heads in Pythia 70M to 2.8B "emerge soon after they have seen 2×10^9 tokens" [Tigges24, section 3.2, Figure 2], "at around step 1,000 out of 143,000" [YinSteinhardt25, section 5.1].
- In-context learning develops "in a narrow window early in training (roughly 2.5 to 5 billion tokens)" in Anthropic's models, alongside induction-head formation [Olsson22, Argument 1].
- In Pythia-70M, many tasks share a phase transition "coinciding with the learning rate warm-up", and many early context neurons are later unlearned [Quirke23, abstract and section 3].
- Pythia's readout reorganizes in "a narrow early window, 0.4-1.1% of the way through Pythia's pretraining" (about steps 570 to 1,570) [He26].
- Networks learn low-order token statistics first [Belrose24].

**Verdict: partly known.** I found no paper that measures how a mid-layer neuron's direct input splits between MLP neurons and attention heads over training. But the crossover falls in a window where several transitions are already documented: warm-up ending, induction heads forming, the readout reorganizing. So the likely reading is that this is one more view of those transitions, not a separate phenomenon. If it's reported, it should be framed that way, with warm-up as a possible confound.

**Searched:** queries 8 and 12 in the [search log](#search-log).

### 4b. The residual-stream mean direction grows with model size

**The repo's finding** (README v6 point 6, exploratory). After LayerNorm normalization, the mean over random positions makes up 26% of the squared norm in 70M's middle layer and 55% in 2.8B's (73% at 2.8B's three-quarters layer).

**Closest prior work**
- Machina & Mercer compute the average cosine similarity between hidden states ("Inter-Sim") at every layer of Pythia 70M, 160M, 410M, 1.4B and 6.9B. 70M and 160M have relatively low values in their middle layers, while 1.4B and 6.9B have high values there, "quite high in the case of 6.9B", then drop at the last layer [MachinaMercer24, section 4.1.1, Figure 1]. Their Eq. 7 rewrites the average pairwise cosine as the squared norm of the summed unit vectors. For vectors of equal norm, as after LayerNorm normalization, that is the repo's measure up to a negligible term.
- Average cosine between random tokens' representations as an anisotropy measure, subtracted as a baseline before comparing similarities, goes back to Ethayarajh, who found GPT-2's upper layers highly anisotropic [Ethayarajh19, sections 3.4 and 4.1].

**Verdict: already published.** The repo should cite [MachinaMercer24] where it mentions the growing mean (README v6 point 6), and not count it as a finding. Centering before comparing similarities is standard practice [Ethayarajh19], so v6's choice to remove the mean is well supported.

### 4c. Neurons repurposed during training

**The repo's finding** (README v1 point 4, exploratory, from reading the top examples at each checkpoint). Neuron 3282's main job goes "bone" to "There / Most" to object nouns to digits inside codes, and its "behind her" facet looks like a leftover from the object-noun phase. Neurons 2918 and 2382 look similar. The README calls repurposing a third way to get a mixed-looking neuron, besides branching and packing.

**Closest prior work**
- *The idea.* Goh et al. guess that conjoined features may be "a vestigial phenomenon from early in training when the model couldn't distinguish between the two concepts in that facet" [Goh21, footnote to "Polysemanticity and Conjoined Neurons"]. The essay's InceptionV1 mixed5a:178 drifts between dome, fire engine and station wagon examples over training [Cammarata26].
- *Changes of role in language-model training.*
  - SAE features in Pythia-410M: "Some features alter their primary semantic association during training" [Xu24, section 3, Figure 2].
  - Token detectors decline as grammatical features rise [Bayazit26, section 7.1, Table 1].
  - Context neurons in Pythia-70M form early and are later unlearned [Quirke23].
- *Evidence for stability, which cuts the other way.*
  - Explanations of a fully trained GPT-3-series model's neurons lose less than 25% of their score when applied at a quarter of training, which the authors read as "a relatively high degree of stability in feature-neuron correspondence" [Bills23, section "Subject model training time"].
  - Universal neurons in GPT-2 small stay stable across checkpoints [Nandan25].
  - In a vision model, Network Dissection finds no "transitions across different concept categories during training" [Bau17, section 3.5, Figure 10].

**Verdict: partly known.** The idea that a neuron's earlier job can survive as a facet was proposed as a guess [Goh21], and feature-level drift during training is documented [Xu24]. The repo adds a few neuron-level examples in a language model, from reading checkpoints by hand. They are anecdotes, and the stability results above suggest repurposing may be uncommon.

**Searched:** queries 7, 8 and 11 in the [search log](#search-log).

## Methods the repo could adopt

1. **Read below the top 50.** Top-activating examples alone are a known source of illusions [Bolukbasi21; Bills23, section "Neurons may represent many features"; Gurnee23, section 5.1]. Foote samples examples above 75% of the maximum activation for this reason [Foote24, section 4.1.2]. Sampling activation quantiles, as autointerp work does [Paulo24], would test whether the readers' disagreement holds below the top.
2. **Score the LLM reader by prediction, not only by a label.** Simulation and detection scoring [Bills23; Paulo24] measure whether a reader's description predicts activations. Multi-description methods give a graded polysemanticity score [Kopf25; Liu26]. Any such score should pass the missing-labels and extra-labels sanity checks [Oikarinen25] and the repo's pooled-neuron control.
3. **Add readers from the neuron's own input.** Neuron embeddings [Foote24] and attribution clustering [Dreyer24] group examples by what drives the neuron, which v2 and v4's upstream drivers already compute. Either would be a fourth reader for the scale sweep, and a check on v6's choice to project the neuron's input direction out.
4. **Control for word form.** Shared word forms inflate polysemanticity metrics [AitHouHwa26]. Since v6's facets often split by token (token purity, README v6 point 7), a check that holds the activating token fixed would separate word-form effects from context.
5. **Test conditional units.** Known-answer benchmarks include units that respond to one concept only in the presence of another [Schwettmann23; Shaham24]. A conditional or unequal pool would be a harder, more realistic positive control than 50/50 pools.
6. **Place training-dynamics observations against known landmarks.** Before reading the MLP-to-attention crossover (4a) as new, compare it with warm-up's end (step 1,430), induction-head formation (about step 1k) [Tigges24] and the readout reorganization [He26]. For repurposing (4c), feature-level tracking across checkpoints [Xu24; Bayazit26] doesn't fix facets at the final checkpoint, which v1 to v4 do.
7. **Cite the anisotropy result** [MachinaMercer24] in README v6 point 6.

## References

- **[AitHouHwa26]** Iyad Ait Hou, Rebecca Hwa (2026). "Polysemanticity or Polysemy? Lexical Identity Confounds Superposition Metrics." arXiv:2604.00443. https://arxiv.org/abs/2604.00443
- **[Bau17]** David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, Antonio Torralba (2017). "Network Dissection: Quantifying Interpretability of Deep Visual Representations." CVPR 2017. arXiv:1704.05796. https://arxiv.org/abs/1704.05796
- **[Bayazit26]** Deniz Bayazit, Aaron Mueller, Antoine Bosselut (2026). "Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining." ACL 2026 (2026.acl-long.60). arXiv:2509.05291. https://arxiv.org/abs/2509.05291
- **[Belrose24]** Nora Belrose, Quintin Pope, Lucia Quirke, Alex Mallen, Xiaoli Fern (2024). "Neural Networks Learn Statistics of Increasing Complexity." ICML 2024. arXiv:2402.04362. https://arxiv.org/abs/2402.04362
- **[Bills23]** Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, William Saunders (2023). "Language models can explain neurons in language models." OpenAI. https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html (sections "Subject model size", "Subject model training time", and "Neurons may represent many features" under "Limitations and caveats")
- **[Bolukbasi21]** Tolga Bolukbasi, Adam Pearce, Ann Yuan, Andy Coenen, Emily Reif, Fernanda Viégas, Martin Wattenberg (2021). "An Interpretability Illusion for BERT." arXiv:2104.07143. https://arxiv.org/abs/2104.07143
- **[Cammarata26]** Nick Cammarata (2026-10-15). "Neural Networks Are Already Clean." Goodfire. https://already-clean.goodfire.com/ (sections "Recursive Generalization" and "Superposition", the figure "A feature takes shape over training", and the Qwen3-32B audit figure in the introduction)
- **[Dravid26]** Amil Dravid, Yasaman Bahri, Alexei A. Efros, Yossi Gandelsman (2026). "Neuron Populations Exhibit Divergent Selectivity with Scale." arXiv:2606.03990. https://arxiv.org/abs/2606.03990
- **[Dreyer24]** Maximilian Dreyer, Erblina Purelku, Johanna Vielhaben, Wojciech Samek, Sebastian Lapuschkin (2024). "PURE: Turning Polysemantic Neurons Into Pure Features by Identifying Relevant Circuits." CVPR 2024 Workshops (XAI4CV). arXiv:2404.06453. https://arxiv.org/abs/2404.06453
- **[Elhage22]** Nelson Elhage, Tristan Hume, Catherine Olsson, et al. (2022). "Softmax Linear Units." Transformer Circuits Thread. https://transformer-circuits.pub/2022/solu/index.html
- **[Ethayarajh19]** Kawin Ethayarajh (2019). "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings." EMNLP-IJCNLP 2019. arXiv:1909.00512. https://arxiv.org/abs/1909.00512
- **[Foote24]** Alex Foote (2024). "Tackling Polysemanticity with Neuron Embeddings." ICML 2024 Mechanistic Interpretability Workshop. arXiv:2411.08166. https://arxiv.org/abs/2411.08166
- **[Goh21]** Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, Chris Olah (2021). "Multimodal Neurons in Artificial Neural Networks." Distill. https://distill.pub/2021/multimodal-neurons/ (section "Feature properties", paragraph "Polysemanticity and Conjoined Neurons" and its footnote)
- **[GuptaKumar25]** Manan Gupta, Dhruv Kumar (2025). "Disentangling Polysemantic Neurons with a Null-Calibrated Polysemanticity Index and Causal Patch Interventions." arXiv:2508.16950. https://arxiv.org/abs/2508.16950
- **[GuptaKumar26]** Manan Gupta, Dhruv Kumar (2026). "DPMI: A Principled Index for Neural Polysemanticity via Dirichlet Process Mixture Modeling." ICML 2026. https://openreview.net/forum?id=AnYCqLSigg (abstract: https://icml.cc/virtual/2026/67442)
- **[Gurnee23]** Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, Dimitris Bertsimas (2023). "Finding Neurons in a Haystack: Case Studies with Sparse Probing." TMLR. arXiv:2305.01610. https://arxiv.org/abs/2305.01610
- **[He26]** Matteo He, William F. Shen, Alex Iacob, Andrej Jovanovic, Xinchi Qiu, Nicholas D. Lane (2026). "Learning to Read Out: Unembedding Dynamics in Language Model Pretraining." arXiv:2610.21025. https://arxiv.org/abs/2610.21025
- **[Kopf25]** Laura Kopf, Nils Feldhus, Kirill Bykov, Philine Lou Bommer, Anna Hedström, Marina M.-C. Höhne, Oliver Eberle (2025). "Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework." NeurIPS 2025. arXiv:2506.15538. https://arxiv.org/abs/2506.15538
- **[Lecomte24]** Victor Lecomte, Kushal Thaman, Rylan Schaeffer, Naomi Bashkansky, Trevor Chow, Sanmi Koyejo (2024). "What Causes Polysemanticity? An Alternative Origin Story of Mixed Selectivity from Incidental Causes." ICLR 2024 Workshop on Representational Alignment (Re-Align). arXiv:2312.03096. https://arxiv.org/abs/2312.03096
- **[Liu26]** Weiqi Liu, Yongliang Miao, Haiyan Zhao, Yanguang Liu, Mengnan Du (2026). "NeuronScope: A Multi-Agent Framework for Explaining Polysemantic Neurons in Language Models." arXiv:2601.03671. https://arxiv.org/abs/2601.03671
- **[MachinaMercer24]** Anemily Machina, Robert Mercer (2024). "Anisotropy is Not Inherent to Transformers." NAACL 2024, pages 4892 to 4907. https://aclanthology.org/2024.naacl-long.274
- **[Nandan25]** Advey Nandan, Cheng-Ting Chou, Amrit Kurakula, Cole Blondin, Kevin Zhu, Vasu Sharma, Sean O'Brien (2025). "Universal Neurons in GPT-2: Emergence, Persistence, and Functional Impact." arXiv:2508.00903. https://arxiv.org/abs/2508.00903
- **[Nguyen16]** Anh Nguyen, Jason Yosinski, Jeff Clune (2016). "Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks." arXiv:1602.03616. https://arxiv.org/abs/1602.03616
- **[Oikarinen25]** Tuomas Oikarinen, Ge Yan, Tsui-Wei Weng (2025). "Evaluating Neuron Explanations: A Unified Framework with Sanity Checks." ICML 2025 (PMLR 267). arXiv:2506.05774. https://arxiv.org/abs/2506.05774
- **[Olsson22]** Catherine Olsson, Nelson Elhage, Neel Nanda, et al. (2022). "In-context Learning and Induction Heads." Transformer Circuits Thread. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html (Argument 1)
- **[Paulo24]** Gonçalo Paulo, Alex Mallen, Caden Juang, Nora Belrose (2024). "Automatically Interpreting Millions of Features in Large Language Models." arXiv:2410.13928. https://arxiv.org/abs/2410.13928
- **[Pythia-config]** EleutherAI. Pythia-410M training configuration, `"warmup": 0.01` and `"train-iters": 143000`. https://github.com/EleutherAI/pythia/blob/main/models/410M/pythia-410m.yml
- **[Quirke23]** Lucia Quirke, Lovis Heindrich, Wes Gurnee, Neel Nanda (2023). "Training Dynamics of Contextual N-Grams in Language Models." ATTRIB workshop at NeurIPS 2023. arXiv:2311.00863. https://arxiv.org/abs/2311.00863
- **[Schwettmann23]** Sarah Schwettmann, Tamar Rott Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, Antonio Torralba (2023). "FIND: A Function Description Benchmark for Evaluating Interpretability Methods." NeurIPS 2023. arXiv:2309.03886. https://arxiv.org/abs/2309.03886
- **[Shaham24]** Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, Antonio Torralba (2024). "A Multimodal Automated Interpretability Agent." ICML 2024. arXiv:2404.14394. https://arxiv.org/abs/2404.14394
- **[Tigges24]** Curt Tigges, Michael Hanna, Qinan Yu, Stella Biderman (2024). "LLM Circuit Analyses Are Consistent Across Training and Scale." NeurIPS 2024. arXiv:2407.10827. https://arxiv.org/abs/2407.10827
- **[Xu24]** Yang Xu, Yi Wang, Hengguan Huang, Hao Wang (2024). "Tracking the Feature Dynamics in LLM Training: A Mechanistic Study." arXiv:2412.17626. https://arxiv.org/abs/2412.17626
- **[YinSteinhardt25]** Kayo Yin, Jacob Steinhardt (2025). "Which Attention Heads Matter for In-Context Learning?" arXiv:2502.14010. https://arxiv.org/abs/2502.14010

## Search log

All searches ran on 2026-10-19 with a web search tool, in "standard" or the more thorough "extended" mode. Every source cited above was then read in its own text (arXiv HTML, the PDF, or the page's source), not taken from a search summary.

1. `polysemanticity index PSI neurons metric top-activating examples` (standard)
2. `synthetic polysemantic neuron positive control merging two neurons top activating examples evaluate polysemanticity detection` (extended)
3. `language model neuron explanation evaluation known mixture of two neurons concatenated activating examples detect polysemantic autointerp power` (extended)
4. `neuron polysemanticity across model scale Pythia different interpretability metrics disagree scale trend` (extended)
5. `agreement between polysemanticity measures LLM judge versus embedding clustering neurons comparison monosemanticity metrics disagree` (extended)
6. `synthetic polysemantic neurons ground truth benchmark interpretability agent FIND MAIA` (standard)
7. `polysemantic neurons formation during training language model checkpoints facets emerge separately neuron repurposed later training` (extended)
8. `tracking individual neurons or features across Pythia training checkpoints emergence drift sparse autoencoder crosscoder` (extended)
9. `"Neural Networks Are Already Clean" Cammarata recursive generalization response` (extended), and `"recursive generalization" neurons Pythia checkpoints test Goodfire essay LessWrong` (standard)
10. `sentence embedding clustering of a neuron's top activating contexts to detect polysemantic neurons language model MiniLM` (extended), and `polysemantic neurons Pythia model size top activating examples clustering` restricted to lesswrong.com, alignmentforum.org, transformer-circuits.pub, blog.eleuther.ai and goodfire.ai (standard)
11. `"conjoined features" OR "conjoined neurons" polysemanticity training dynamics follow-up` (standard)
12. `attention heads versus MLP neurons share of contribution changes over training early checkpoints language model direct attribution Pythia` (extended)
13. `anisotropy Pythia model size training checkpoints residual stream mean cosine similarity` (standard), and `Machina Mercer "Anisotropy is not inherent to transformers" Pythia NAACL 2024` (standard)

Targeted lookups of papers already known to be relevant (Bills23, Gurnee23, Lecomte24, Oikarinen25, Foote24, Dreyer24, Nguyen16, Xu24, Bayazit26, Tigges24, Belrose24, Bolukbasi21) are not listed. The searches cover English-language sources indexed by the search tool. Papers behind paywalls, talks and private write-ups would be missed.
