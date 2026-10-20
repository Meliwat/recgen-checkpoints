import numpy as np
import pytest

from recgen import analyze_v22
from recgen.analyze_v22 import comparison, paired_flags, sample_eligibility
from recgen.rubric_v19 import MODELS
from recgen.rubric_v22 import CORPUS_PART


def test_paired_flags_match_by_id_and_skip_units_missing_a_label():
    owt = [{"id": "a", "llm": True}, {"id": "b", "llm": None}, {"id": "c", "llm": False}, {"id": "d", "llm": True}]
    pile = [{"id": "d", "llm": False}, {"id": "c", "llm": False}, {"id": "b", "llm": True}, {"id": "a", "llm": None}]
    o, p = paired_flags(owt, pile, "id", "llm")
    assert o.tolist() == [0.0, 1.0] and p.tolist() == [0.0, 0.0]  # c and d
    with pytest.raises(AssertionError):
        paired_flags(owt, pile[:3], "id", "llm")


def test_sample_eligibility_counts_v19s_neurons_that_are_dead_or_dense_here(tmp_path, monkeypatch):
    window_max = np.full((100, 6), 0.1, np.float16)
    window_max[0] = 5.0
    window_max[:, 1] = 0.01
    window_max[0, 1] = 0.1  # near-dead
    window_max[:80, 2] = 4.0  # dense
    np.savez(tmp_path / "m.npz", max=window_max)
    monkeypatch.setattr(analyze_v22, "maxima_path", lambda model: tmp_path / "m.npz")
    neurons = [{"neuron": n, "peakedness": float("nan") if n == 1 else 2.0} for n in (1, 2, 3)]
    out = sample_eligibility("gpt2", neurons)
    assert out == {"sampled_near_dead": 1, "sampled_dense": 1, "sampled_ineligible": 2, "sampled_peak_not_positive": 0,
                   "sampled_peakedness_undefined": 1}


def rows(group: str, flags: list) -> list[dict]:
    return [{"id": f"{group}{i}", "group": group, "minilm": False, "network": bool(i % 2), "llm": f,
             "llm_label": None if f is None else "C" if f else "A"} for i, f in enumerate(flags)]


def test_comparison_pairs_the_same_units_in_both_corpora():
    owt_real, pile_real = [True] * 30 + [False] * 70, [True] * 10 + [False] * 90  # +20 points on real neurons
    units = {m: rows("real", owt_real) + rows("pooled 50/50", [True] * 30 + [False] * 20) for m in MODELS}
    neurons = {m: [{"neuron": i, "network": 1.0, "minilm": 0.0, "coherence": 80.0} for i in range(5)] for m in MODELS}
    v19 = {"units": {m: rows("real", pile_real) + rows("pooled 50/50", [True] * 20 + [False] * 30) for m in MODELS},
           "neurons": {m: [{"neuron": i, "network": 0.0, "minilm": 0.0, "coherence": 90.0} for i in range(5)]
                       for m in MODELS}}
    labels = {f"{size}:real::{i}": {"label": "C" if i < 60 else "A"} for size in ("160m", "410m", "1b") for i in range(100)}
    out = comparison(units, neurons, v19, {"first": labels})
    c = out["c"]
    assert abs(c["change"] - 0.2) < 1e-12 and abs(c["pythia_size_matched"] - 0.6) < 1e-12 and c["verdict"] == CORPUS_PART
    assert abs(c["share_of_gap_closed"] - 0.4) < 1e-12  # 20 of the 50 points to Pythia's rate
    assert c["by_model"]["gpt2"]["flagged_only_on_openwebtext"] == 20 and c["by_model"]["gpt2"]["flagged_only_on_pile"] == 0
    assert abs(out["d"]["llm"]["change"] - (0.2 - 0.2)) < 1e-12  # pools and real units both rose 20 points
    assert out["secondary"]["paired_rates"]["all_sampled"]["network"]["change"] == 1.0
    assert out["secondary"]["coherence_median"]["gpt2"] == {"openwebtext": 80.0, "pile": 90.0}
    assert out["secondary"]["haiku_labels_real"]["gpt2"]["pile"] == {"C": 10, "A": 90}
