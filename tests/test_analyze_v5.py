import numpy as np

from recgen.analyze_v5 import decisions, layer_stats, neuron_facets
from recgen.rubric_v5 import DEPTHS, FITS, MODELS, NO_TREND


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_neuron_facets_counts_facets_and_their_least_related_similarity():
    rng = np.random.default_rng(0)
    a = unit(np.array([1.0, 0, 0]) + 0.05 * rng.normal(size=(30, 3)))
    b = unit(np.array([0, 1.0, 0]) + 0.05 * rng.normal(size=(20, 3)))
    examples = [(w, 40, 1.0) for w in range(50)]
    vectors = {(w, 40): v for w, v in zip(range(50), np.vstack([a, b]))}
    r = neuron_facets(examples, vectors)
    assert r["n_facets"] == 2 and abs(r["least_related_similarity"]) < 0.1
    assert sorted(len(f) for f in r["facets"]) == [20, 30]


def test_layer_stats_shares():
    neurons = {0: {"n_facets": 2, "least_related_similarity": 0.0}, 1: {"n_facets": 2, "least_related_similarity": 0.5},
               2: {"n_facets": 1, "least_related_similarity": None}, 3: {"n_facets": 0, "least_related_similarity": None}}
    s = layer_stats("70m", "middle", {"layer": 3, "n_layers": 6, "width": 2048, "eligible": 4}, neurons)
    assert s["unrelated_mixed_share"] == 0.25 and s["multi_facet_share"] == 0.5
    assert s["median_least_related_similarity"] == 0.25


def test_decisions_flag_the_primary_measure_and_judge_each_depth():
    stats = []
    for i, model in enumerate(MODELS):
        for depth in DEPTHS:
            share = 0.30 - 0.04 * i if depth == "middle" else 0.2
            stats.append({"model": model, "depth": depth, "unrelated_mixed_share": share,
                          "unrelated_mixed_interval": (share - 0.01, share + 0.01),
                          "multi_facet_share": 0.2, "multi_facet_interval": (0.19, 0.21)})
    out = decisions(stats)
    assert out["middle: unrelated_mixed"]["verdict"] == FITS and out["middle: unrelated_mixed"]["primary"]
    assert out["middle: multi_facet"]["verdict"] == NO_TREND and not out["middle: multi_facet"]["primary"]


def test_shared_token_rates_compare_a_neurons_unrelated_facets_with_other_neurons_facets():
    from recgen.analyze_v5 import shared_token_rates
    from recgen.facets import tokenizer

    ids = {w: tokenizer().encode(w)[0] for w in (" See", " find", " cat", " Id", " it", " dog")}
    corpus = np.zeros((6, 40), dtype=np.int64)
    # Neuron k's two facets both fire on the same token (a[k]) but are followed by different tokens.
    a = [" See", " find", " cat"]
    nexts = [(" Id", " it"), (" it", " dog"), (" dog", " Id")]
    neurons = {}
    for k in range(3):
        facets = []
        for f in range(2):
            w = 2 * k + f
            corpus[w, 33], corpus[w, 34] = ids[a[k]], ids[nexts[k][f]]
            facets.append([(w, 33, 1.0)] * 5)
        neurons[k] = {"n_facets": 2, "least_related_similarity": 0.0, "least_related_pair": (0, 1), "facets": facets}
    rates = shared_token_rates(neurons, corpus, np.random.default_rng(0))
    assert rates["neurons"] == 3
    assert rates["least_related_pair"] == {"same_activating_token": 1.0, "same_next_token": 0.0}
    assert rates["null"]["same_activating_token"] == 0.0
