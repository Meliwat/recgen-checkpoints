import numpy as np
import pytest

from recgen.rubric_v3 import (
    FACET_MAX, HARMFUL_INTERFERENCE, HARMLESS, HARMLESS_INTERFERENCE, HARMS, HELPS, MIN_EXAMPLES, MIN_RELIABILITY,
    NO_USE, NOT_SPECIFIC, ONE_JOB, TOO_FEW, TWO_USES, Signatures, enlarge, least_related_pair, pair_label, scorable,
    shared_job, spearman_brown, use,
)


def test_scorable_needs_five_next_tokens_inside_the_window():
    assert scorable(506, 512)
    assert not scorable(507, 512)


def test_enlarge_keeps_originals_and_adds_nearest_most_similar_first():
    similarities = np.array([
        [0.9, 0.1],  # original of facet 0
        [0.1, 0.9],  # original of facet 1
        [0.5, 0.6],  # nearest to facet 1
        [0.45, 0.0],
        [0.7, 0.1],
        [0.3, 0.2],  # below the floor
        [0.8, 0.0],  # unscorable
    ])
    keep = np.array([True] * 6 + [False])
    assert enlarge(similarities, [[0], [1]], keep, assign_min=0.4) == [[0, 4, 3], [1, 2]]
    assert enlarge(similarities, [[0], [1]], keep, assign_min=0.4, facet_max=2) == [[0, 4], [1, 2]]


def test_enlarge_drops_unscorable_originals():
    similarities = np.array([[0.9], [0.8], [0.7]])
    assert enlarge(similarities, [[0, 1]], np.array([False, True, True])) == [[1, 2]]


def test_enlarge_never_exceeds_the_cap():
    similarities = np.random.default_rng(0).uniform(0.5, 1, size=(100, 1))
    assert len(enlarge(similarities, [[0]], np.ones(100, bool))[0]) == FACET_MAX


def test_use_needs_both_significance_and_size():
    rng = np.random.default_rng(0)
    assert use(0.05 + 0.01 * rng.standard_normal(20)) == HELPS
    assert use(-0.05 + 0.01 * rng.standard_normal(20)) == HARMS
    assert use(0.003 + 0.001 * rng.standard_normal(20)) == HARMLESS  # consistent but below the floor
    assert use(0.003 + 0.001 * rng.standard_normal(20), min_help=0.0) == HELPS
    assert use(0.05 * rng.standard_normal(20)) == HARMLESS  # noise
    assert use(np.zeros(12)) == HARMLESS


def test_specific_signatures_match_direct_computation():
    rng = np.random.default_rng(1)
    effects, generic = rng.standard_normal((6, 30)), rng.standard_normal(30)
    sig = Signatures(effects @ effects.T, effects @ generic, generic @ generic)
    unit = generic / np.linalg.norm(generic)
    specific = lambda rows: effects[rows].mean(axis=0) - (effects[rows].mean(axis=0) @ unit) * unit
    cos = lambda x, y: x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
    a, b = [0, 1, 2], [3, 4, 5]
    np.testing.assert_allclose(sig.specific_cosine(a, b), cos(specific(a), specific(b)))
    np.testing.assert_allclose(sig.raw_cosine(a, b), cos(effects[a].mean(axis=0), effects[b].mean(axis=0)))
    np.testing.assert_allclose(sig.generic_share(a), cos(effects[a].mean(axis=0), generic) ** 2)


def test_spearman_brown_steps_half_agreement_up_and_floors_at_zero():
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(-0.2) == 0.0


def synthetic(job_b: str, noise_b: float = 1.0, generic_scale: float = 3.0, seed: int = 0, n: int = 30, d: int = 400):
    """Effects = a generic push of varying size shared by every context + the facet's job +
    noise. The null has only the generic push and its own context noise.
    job_b: "same" as A's, "other", or "none"."""
    rng = np.random.default_rng(seed)
    generic, job_a, other = (rng.standard_normal(d) for _ in range(3))
    job = {"same": job_a, "other": other, "none": np.zeros(d)}[job_b]
    scale = lambda k: generic_scale * rng.uniform(0.5, 1.5, size=(k, 1)) * generic
    a = scale(n) + job_a + rng.standard_normal((n, d))
    b = scale(n) + job + noise_b * rng.standard_normal((n, d))
    push = (scale(1000) + rng.standard_normal((1000, d))).mean(axis=0)
    effects = np.vstack([a, b])
    sig = Signatures(effects @ effects.T, effects @ push, push @ push)
    rows_a, rows_b = list(range(n)), list(range(n, 2 * n))
    rel_a, rel_b = sig.reliability(rows_a, rng), sig.reliability(rows_b, rng)
    return {"raw": sig.raw_cosine(rows_a, rows_b), "rel_b": rel_b,
            "score": shared_job(sig.specific_cosine(rows_a, rows_b), rel_a, rel_b)}


@pytest.mark.parametrize("seed", range(3))
def test_shared_job_is_near_one_for_the_same_job_even_when_one_facet_is_noisier(seed):
    result = synthetic("same", noise_b=2.0, seed=seed)
    assert result["rel_b"] >= MIN_RELIABILITY
    assert 0.8 < result["score"] < 1.2


@pytest.mark.parametrize("seed", range(3))
def test_shared_job_is_near_zero_for_unrelated_jobs_though_raw_signatures_look_alike(seed):
    result = synthetic("other", seed=seed)
    assert result["raw"] > 0.8  # the shared push makes the signatures look alike by construction
    assert abs(result["score"]) < 0.2


def test_facet_with_only_the_generic_push_has_no_reliable_specific_signature():
    assert synthetic("none")["rel_b"] < MIN_RELIABILITY


def test_pair_label_order():
    n = MIN_EXAMPLES
    assert pair_label(n - 1, n, HELPS, HELPS, 1, 1, 1.0) == TOO_FEW
    assert pair_label(n, n, HELPS, HELPS, 1, 1, 0.5) == ONE_JOB
    assert pair_label(n, n, HELPS, HELPS, 1, 1, 0.49) == TWO_USES
    assert pair_label(n, n, HELPS, HELPS, 1, MIN_RELIABILITY - 0.01, 1.0) == NOT_SPECIFIC
    assert pair_label(n, n, HELPS, HARMLESS, 1, 1, 1.0) == HARMLESS_INTERFERENCE
    assert pair_label(n, n, HARMS, HELPS, 1, 1, 1.0) == HARMFUL_INTERFERENCE
    assert pair_label(n, n, HARMS, HARMLESS, 1, 1, 1.0) == NO_USE


def test_least_related_pair_has_the_lowest_similarity():
    assert least_related_pair({"0-1": 0.4, "0-2": -0.1, "1-2": 0.0}) == (0, 2)
