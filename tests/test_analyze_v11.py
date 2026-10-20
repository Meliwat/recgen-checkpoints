from recgen.analyze_v7 import GROUPS, model_stats, verdicts
from recgen.analyze_v11 import depth_summary, excess_by_reader
from recgen.rubric_v5 import MODELS
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
