import numpy as np

from recgen.analyze_v9 import analyze, bin_rates
from recgen.rubric_v5 import MODELS


def union(minority: int, flagged: bool) -> dict:
    return {"minority": minority, "minilm": flagged, "network": flagged, "llm": flagged}


def test_bin_rates_rise_when_balanced_unions_are_flagged_more():
    unions = [union(1, i < 2) for i in range(20)] + [union(10, i < 8) for i in range(20)] + [union(20, i < 15) for i in range(20)]
    out = bin_rates(unions, "network", np.random.default_rng(0))
    assert [b["share"] for b in out["bins"]] == [0.1, 0.4, 0.75]
    assert abs(out["top_minus_bottom"] - 0.65) < 1e-9 and out["interval"][0] > 0


def test_analyze_reports_union_minus_real_for_every_reader():
    per_model, v7_units = {}, {}
    for m in MODELS:
        unions = [union(20, i < 12) for i in range(20)]
        real = [{"minilm": i < 4, "network": i < 4, "llm": i < 4} for i in range(20)]
        per_model[m] = {"unions": unions, "real": real, "replaced": 0,
                        "sensitivity": {how: {"dropped": 0, "units": unions} for how in ("peak", "top-50 level")}}
        v7_units[m] = [{"group": "pooled 50/50", "minilm": i < 14, "network": i < 14, "llm": i < 14} for i in range(20)]
    answers = {"first": {"a": {"label": "C"}}, "retest": {"a": {"label": "C"}}}
    summary = analyze(per_model, v7_units, answers)
    test = summary["tests"]["llm"]
    assert abs(test["union_minus_real"]["mean_over_models"] - 0.4) < 1e-9 and test["union_minus_real"]["detects_unions"]
    assert abs(test["v7_pools_minus_unions"]["mean_over_models"] - 0.1) < 1e-9
