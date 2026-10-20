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


def test_quartile_rates_and_partial_spearman():
    from recgen.analyze_v9 import partial_spearman, quartile_rates

    peak = np.arange(100.0)
    out = quartile_rates(peak, (peak < 75).astype(float))
    assert out["by_quartile"] == [1.0, 1.0, 1.0, 0.0] and out["spearman"] < 0
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x, y = z + 0.1 * rng.normal(size=500), z + 0.1 * rng.normal(size=500)
    assert abs(partial_spearman(x, y, [z])) < 0.2 < np.corrcoef(x, y)[0, 1]


def test_token_purity_counts_the_most_common_activating_token():
    from recgen.analyze_v9 import token_purity

    corpus = np.zeros((60, 4), int)
    corpus[:, 1] = 9  # a token only the weakest windows would show, at position 1
    corpus[30:, 2] = 5  # 30 of the 50 strongest windows fire on token 5 at position 2
    corpus[10:30, 2] = np.arange(20) + 100  # the other 20 on distinct tokens
    window_max = np.zeros((60, 1), np.float16)
    window_max[:, 0] = np.arange(60)  # windows 10 to 59 are the 50 strongest
    window_pos = np.full((60, 1), 2)
    window_pos[:10, 0] = 1
    assert token_purity(window_max, window_pos, 0, corpus) == 30 / 50
