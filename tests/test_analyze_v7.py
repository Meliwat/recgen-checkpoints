from recgen.analyze_v7 import GROUPS, model_stats, retest_stats, verdicts
from recgen.rubric_v5 import MODELS


def rows_with(rates: dict[str, float], llm_missing: int = 0) -> list[dict]:
    """100 synthetic units per group; every reader flags the given share of each group."""
    rows = []
    for g in GROUPS:
        for i in range(100):
            flag = i < 100 * rates[g]
            rows.append({"group": g, "minilm": flag, "network": flag, "llm": flag, "llm_label": "C" if flag else "A"})
    for r in rows[:llm_missing]:
        r["llm"], r["llm_label"] = None, None
    return rows


SEPARATING = {"real": 0.1, "pooled 50/50": 0.7, "pooled 70/30": 0.5, "pooled 90/10": 0.2, "related": 0.1}
BLIND = {"real": 0.1, "pooled 50/50": 0.15, "pooled 70/30": 0.1, "pooled 90/10": 0.1, "related": 0.1}


def test_a_reader_that_separates_pools_from_real_neurons_has_power_and_is_specific():
    s = model_stats("410m", rows_with(SEPARATING))
    assert s["power"]["llm"]["has_power"] and abs(s["power"]["llm"]["excess"] - 0.6) < 1e-9
    assert s["specificity"]["llm"]["specific"]
    assert s["rates"]["network"]["pooled 70/30"]["share"] == 0.5


def test_a_blind_reader_has_no_power():
    assert not model_stats("410m", rows_with(BLIND))["power"]["minilm"]["has_power"]


def test_units_without_a_label_drop_out_of_the_llm_rates_only():
    s = model_stats("410m", rows_with(SEPARATING, llm_missing=10))
    assert s["llm_missing"] == 10 and s["rates"]["llm"]["real"]["n"] == 90 and s["rates"]["minilm"]["real"]["n"] == 100


def test_verdicts_need_power_in_five_of_six_models():
    stats = [model_stats(m, rows_with(SEPARATING if i < 4 else BLIND)) for i, m in enumerate(MODELS)]
    out = verdicts(stats)
    assert out["llm"]["models_with_power"] == 4 and not out["llm"]["power"]


def test_retest_compares_labels_and_flags():
    answers = {"first": {"a": {"label": "C"}, "b": {"label": "A"}, "c": {"label": "B"}},
               "retest": {"a": {"label": "C"}, "b": {"label": "B"}, "c": {"label": None}}}
    r = retest_stats(answers)
    assert r["units"] == 2 and r["same_label"] == 0.5 and r["same_flag"] == 1.0


def test_max_kappa_is_reached_when_one_readers_flags_are_a_subset_of_the_others():
    import numpy as np

    from recgen.analyze_v7 import max_kappa
    from recgen.rubric_v6 import cohen_kappa

    a = np.array([True] * 5 + [False] * 95)  # 5% flagged
    b = np.array([True] * 50 + [False] * 50)  # 50% flagged, including all of a's
    assert abs(max_kappa(0.05, 0.5) - cohen_kappa(a, b)) < 1e-9
    assert max_kappa(0.3, 0.3) == 1.0


def test_pass_probability_rises_with_the_true_excess():
    import numpy as np

    from recgen.analyze_v7 import pass_probability

    rng = np.random.default_rng(0)
    assert pass_probability(0.0, rng) < 0.05 and pass_probability(0.5, rng) > 0.95
