import pytest

from recgen.analyze_v7 import GROUPS, exploratory_stats, model_stats, verdicts
from recgen.analyze_v11 import depth_summary, excess_by_reader, exploratory, with_llm_labels
from recgen.rubric_v5 import FITS, MODELS, NO_TREND
from recgen.rubric_v11 import READER_DEPENDENT

SEPARATING = {"real": 0.1, "pooled 50/50": 0.7, "pooled 70/30": 0.5, "pooled 90/10": 0.2, "related": 0.1}


def rows_with(rates: dict[str, dict[str, float]], llm_missing: int = 0) -> list[dict]:
    """100 synthetic units per group; each reader flags its own share of each group."""
    rows = []
    for g in GROUPS:
        for i in range(100):
            flags = {reader: i < 100 * r[g] for reader, r in rates.items()}
            rows.append({"group": g, **flags, "llm_label": "C" if flags["llm"] else "A"})
    for r in [r for r in rows if r["group"] == "pooled 50/50"][:llm_missing]:
        r["llm"], r["llm_label"] = None, None
    return rows


def test_excess_is_pooled_minus_real_for_each_reader_and_skips_missing_labels():
    blind = {**SEPARATING, "pooled 50/50": 0.1}
    rows = {m: rows_with({"minilm": blind, "network": SEPARATING, "llm": SEPARATING}, llm_missing=40) for m in MODELS}
    out = excess_by_reader(rows, "quarter")
    assert abs(out["minilm"]["excess"]) < 1e-9 and abs(out["network"]["excess"] - 0.6) < 1e-9
    # The first 40 of the 50/50 pools (all flagged) lose their label; 30 of the 60 left are flagged.
    assert abs(out["llm"]["excess"] - (30 / 60 - 0.1)) < 1e-9


def test_depth_summary_reports_excess_trend_and_reader_dependence():
    flat = {**SEPARATING, "real": 0.4}
    rows = {}
    for i, m in enumerate(MODELS):
        # MiniLM's real rate falls with size and it never separates pools; the others are flat.
        minilm = {g: 0.3 - 0.05 * i for g in GROUPS}
        rows[m] = rows_with({"minilm": minilm, "network": flat, "llm": flat})
    stats = [model_stats(m, rows[m]) for m in MODELS]
    out = depth_summary("quarter", stats, excess_by_reader(rows, "quarter"), verdicts(stats))
    assert out["verdicts"]["minilm"]["scale"] == "fits the essay"
    assert out["reader_dependence"]["outcome"] == READER_DEPENDENT
    assert out["reader_dependence"]["sensitive"] == ["network", "llm"]
    assert out["excess_per_model"]["network"] == [0.3] * 6 and out["real_rates"]["minilm"][0] == 0.3


def test_v7s_checks_run_once_units_without_a_label_are_left_out():
    rows = {m: rows_with({"minilm": SEPARATING, "network": SEPARATING, "llm": SEPARATING}) for m in MODELS}
    for model_rows in rows.values():
        for i, r in enumerate(model_rows):
            r["shared_window"] = i % 2 == 0
    rows["2.8b"][0]["llm"], rows["2.8b"][0]["llm_label"] = None, None  # a real neuron the language model refused
    answers = {"first": {"70m:real::1": {"label": "C"}, "70m:real::2": {"label": "A"}},
               "retest": {"70m:real::1": {"label": "C"}, "70m:real::2": {"label": "A"}}}
    with pytest.raises(TypeError):
        exploratory_stats(rows, answers)
    labeled = with_llm_labels(rows)
    assert len(labeled["2.8b"]) == len(rows["2.8b"]) - 1 and len(labeled["70m"]) == len(rows["70m"])
    assert set(exploratory_stats(labeled, answers)["kappa_ceiling"]) == set(MODELS)


def test_exploratory_reads_one_job_shares_and_full_sample_verdicts():
    flat = {**SEPARATING, "real": 0.4}
    rows = {m: rows_with({"minilm": flat, "network": flat, "llm": flat}) for m in MODELS}
    stats = [model_stats(m, rows[m]) for m in MODELS]
    stats[0]["llm_labels"]["real"] = {"A": 5, "C": 4, "null": 1}  # a missing label, as JSON stores it
    by_depth = {"quarter": depth_summary("quarter", stats, excess_by_reader(rows, "quarter"), verdicts(stats))}
    # On its full sample the network reader falls with size; Haiku sees no trend here.
    full = {"quarter: unrelated_mixed": {"v5": {"verdict": NO_TREND}, "v6": {"verdict": FITS}}}
    out = exploratory({"quarter": stats}, by_depth, full)
    assert out["llm_one_job_share_real"]["quarter"][0] == round(5 / 9, 3)
    dependence = out["reader_dependence_with_full_samples"]["quarter"]
    assert dependence["verdicts"]["network"] == FITS and dependence["outcome"] == READER_DEPENDENT
