import hashlib
import json

import numpy as np
import pytest

import recgen.review_v19 as review
from recgen.rubric_v19 import MODELS


def reading(label, sha="h"):
    return {"label": label, "prompt_sha256": sha, "input_tokens": 100}


def test_flag_rate_counts_unrelated_jobs_over_labeled_readings():
    assert review.flag_rate(["C", "A", None, "B"]) == pytest.approx(1 / 3)


def test_size_matched_pairs_each_gpt2_size_with_the_nearest_pythia_size():
    v19 = {"first": {f"{m}:real::{i}": reading("C" if i < k else "A") for m, k in zip(MODELS, (1, 2, 3)) for i in range(10)}}
    v19["first"]["gpt2:pooled:50/50:1-2"] = reading("C")
    v7 = {"first": {f"{m}:real::{i}": reading("C" if i < 5 else "A") for m in review.PYTHIA_MODELS for i in range(10)}}
    out = review.size_matched(v19, v7)
    assert out["gpt2"] == {"gpt2": 0.1, "gpt2-medium": 0.2, "gpt2-large": 0.3}
    assert list(out["pythia_size_matched"]) == ["160m", "410m", "1b"] and out["mean"]["pythia_size_matched"] == 0.5
    assert out["mean"]["gpt2"] == pytest.approx(0.2)


def test_retest_kappa_pairs_cached_readings_and_requires_the_same_prompt():
    first = {"a.json": reading("C"), "b.json": reading("A"), "c.json": reading("C"), "d.json": reading("A")}
    out = review.retest_kappa(first, {"a.json": reading("C"), "b.json": reading("A")})
    assert out["units"] == 2 and out["same_flag"] == 1 and out["flag_kappa"] == pytest.approx(1)
    with pytest.raises(AssertionError):
        review.retest_kappa(first, {"a.json": reading("C", sha="other")})


def test_near_in_time_reads_v11s_cache_by_depth(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "V11_DIR", tmp_path)
    (tmp_path / "llm").mkdir()
    (tmp_path / "llm_retest").mkdir()
    for name, label in [("70m_quarter_real__1", "C"), ("70m_quarter_real__2", "A"), ("70m_three_quarters_real__1", "C"),
                        ("70m_quarter_pooled_50-50_1-2", "C")]:
        (tmp_path / "llm" / f"{name}.json").write_text(json.dumps(reading(label)))
    (tmp_path / "llm_retest" / "70m_quarter_real__1.json").write_text(json.dumps(reading("C")))
    out = review.near_in_time({"70m:fresh::1": reading("C"), "70m:fresh::2": reading("B")})
    assert out["v12_fresh_middle"] == {"readings": 2, "flagged": 0.5}
    assert out["v11"]["quarter"] == {"readings": 2, "flagged": 0.5} and out["v11"]["three_quarters"]["flagged"] == 1
    assert out["v11"]["retest"]["units"] == 1


def test_near_in_time_without_v11s_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "V11_DIR", tmp_path)
    assert review.near_in_time({"70m:fresh::1": reading("C")})["v11"] is None


def test_retest_composition_counts_real_neurons_and_first_reading_flags():
    answers = {"first": {"gpt2:real::1": reading("C"), "gpt2:pooled:50/50:1-2": reading("A"), "gpt2:real::3": reading("A")},
               "retest": {"gpt2:real::1": reading("A"), "gpt2:pooled:50/50:1-2": reading("A"), "gpt2:real::3": reading(None)}}
    assert review.retest_composition(answers) == {"units": 2, "real_neurons": 1, "flagged_at_first_reading": 0.5}


def test_example_stats_find_repeated_contexts_and_the_commonest_token():
    corpus = np.arange(40 * 64).reshape(40, 64) % 997
    corpus[1] = corpus[0]  # window 1 repeats window 0's text
    keys = np.array([[0, 20], [1, 20], [2, 30], [3, 30]])
    corpus[2, 30] = corpus[3, 30] = corpus[0, 20]  # three examples share a firing token
    units = [{"kind": "real", "keys": keys}, {"kind": "pooled", "keys": keys[:1]}]
    out = review.example_stats(units, corpus, lambda ids: f" Tok{int(ids[0])}")
    assert out == {"duplicated_context_share": 0.5, "median_top_token_share": 1.0}


def test_prompts_reproduce_compares_rebuilt_prompts_with_stored_hashes(monkeypatch):
    monkeypatch.setattr(review, "gpt2_prompt", lambda unit, corpus: f"prompt for {unit['id']}")
    digest = hashlib.sha256(b"prompt for u1").hexdigest()
    answers = {"first": {"u1": reading("A", digest), "u2": reading("A", "stale")}, "retest": {"u1": reading("C", digest)}}
    out = review.prompts_reproduce({"gpt2": [{"id": "u1"}, {"id": "u2"}]}, None, answers)
    assert out == {"readings": 3, "matching_prompt_hash": 2}


def test_largest_coordinate_finds_an_outsized_dimension():
    vectors = np.random.default_rng(0).normal(size=(100, 8))
    vectors[:90, 5] = 30
    out = review.largest_coordinate(vectors)
    assert out["dimension"] == 5 and out["share_of_positions"] >= 0.9 and out["median_largest"] == 30
    assert out["median_coordinate"] < 2


def test_purity_ties_flags_neurons_whose_top_50_differ_at_a_tie(tmp_path, monkeypatch):
    windows, positions = 120, 64
    window_max = np.tile(np.linspace(10, 1, windows)[:, None], (1, 2)).astype(np.float16)
    window_max[49:51, 1] = window_max[49, 1]  # neuron 1 ties at ranks 50 and 51
    pos = np.full((windows, 2), 40, np.int16)
    path = tmp_path / "maxima.npz"
    np.savez(path, max=window_max, pos=pos)
    corpus = np.zeros((windows, positions), np.int64)
    corpus[50, 40] = 7  # the window the readers include at the tie carries a different token
    readers = np.array([[[w, 40] for w in range(50)], [[w, 40] for w in list(range(49)) + [50]]])
    monkeypatch.setattr(review, "maxima_path", lambda model: path)
    monkeypatch.setattr(review, "sampled_top_examples", lambda model: (np.array([0, 1]), readers, None))
    out = review.purity_ties("gpt2", [{"token_purity": 1.0}, {"token_purity": 1.0}], corpus)
    assert out == {"neurons_differing": 0.5, "largest_purity_change": pytest.approx(0.02)}
