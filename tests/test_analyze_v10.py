import numpy as np

from recgen.analyze_v7 import GROUPS
from recgen.analyze_v10 import (
    agreement, as_llm, label_shares, main_comparison, reader_excess, retest, retest_pairs, retest_versus_haiku,
    v7_tests,
)
from recgen.rubric_v5 import MODELS
from recgen.rubric_v10 import BETTER, NO_DIFFERENCE, multiple_jobs

SEPARATING = {"real": 0.1, "pooled 50/50": 0.8, "pooled 70/30": 0.5, "pooled 90/10": 0.2, "related": 0.1}
WEAK = {"real": 0.3, "pooled 50/50": 0.5, "pooled 70/30": 0.4, "pooled 90/10": 0.3, "related": 0.3}


def rows_with(rates: dict[str, dict[str, float]], n: int = 100) -> dict:
    """n synthetic units per group and model; each reader answers C for the given share of a group, A otherwise."""
    rows = {}
    for m in MODELS:
        model_rows = []
        for g in GROUPS:
            for i in range(n):
                row = {"id": f"{m}:{g}:{i}", "group": g}
                for reader, by_group in rates.items():
                    flag = i < n * by_group[g]
                    row[reader], row[f"{reader}_label"] = flag, "C" if flag else "A"
                model_rows.append(row)
        rows[m] = model_rows
    return rows


def readers(haiku, sonnet, opus=WEAK):
    return {"minilm": WEAK, "network": WEAK, "haiku": haiku, "sonnet": sonnet, "opus": opus}


def test_as_llm_puts_a_reader_in_v7s_slot():
    row = rows_with(readers(WEAK, SEPARATING))["70m"][0]
    moved = as_llm([row], "sonnet")[0]
    assert moved["llm"] == row["sonnet"] and moved["llm_label"] == row["sonnet_label"]


def test_v7_tests_run_per_reader_and_skip_a_reader_missing_a_whole_group():
    rows = rows_with(readers(WEAK, SEPARATING))
    tests = v7_tests(rows, "sonnet")
    assert tests["verdicts"]["models_with_power"] == 6 and tests["per_model"]["410m"]["rates"]["pooled 70/30"]["share"] == 0.5
    for r in rows["1b"]:
        if r["group"] == "related":
            r["sonnet"], r["sonnet_label"] = None, None
    assert v7_tests(rows, "sonnet") is None


def test_reader_excess_and_the_main_comparison():
    rows = rows_with(readers(WEAK, SEPARATING))
    assert abs(reader_excess(rows, "sonnet", 0)["excess"] - 0.7) < 1e-9
    out = main_comparison(rows, "sonnet", 0)
    assert abs(out["difference"] - 0.5) < 1e-9 and out["verdict"] == BETTER and out["units"] == 6 * 200
    assert main_comparison(rows, "opus", 0)["verdict"] == NO_DIFFERENCE


def test_main_comparison_pairs_only_units_both_readers_labeled():
    rows = rows_with(readers(WEAK, SEPARATING))
    for r in rows["70m"][:10]:
        r["sonnet"], r["sonnet_label"] = None, None
    assert main_comparison(rows, "sonnet", 0)["units"] == 6 * 200 - 10
    for m in MODELS:
        for r in rows[m]:
            if r["group"] == "real":
                r["opus"], r["opus_label"] = None, None
    assert main_comparison(rows, "opus", 0)["verdict"] == "not run"


def test_the_secondary_flag_counts_related_jobs_answers():
    rows = rows_with(readers(WEAK, WEAK))
    for m in MODELS:
        for r in rows[m]:
            if r["group"] == "pooled 50/50" and r["sonnet_label"] == "A":
                r["sonnet_label"] = "B"  # every 50/50 pool now has two or more jobs
    assert abs(reader_excess(rows, "sonnet", 0, multiple_jobs)["excess"] - 0.7) < 1e-9
    assert abs(reader_excess(rows, "sonnet", 0)["excess"] - 0.2) < 1e-9


def test_retest_drops_unlabeled_readings_and_compares_with_haiku_on_shared_units():
    haiku = {"first": {"a": {"label": "C"}, "b": {"label": "A"}, "c": {"label": "C"}, "70m:real::1": {"label": "A"}},
             "retest": {"a": {"label": "A"}, "b": {"label": "A"}, "c": {"label": "C"}, "70m:real::1": {"label": "C"}}}
    sonnet = {"first": {"a": {"label": "C"}, "b": {"label": "A"}, "c": {"label": None}, "70m:real::1": {"label": "A"}},
              "retest": {"a": {"label": "C"}, "b": {"label": "A"}, "c": {"label": "C"}, "70m:real::1": {"label": "A"}}}
    assert set(retest_pairs(sonnet)) == {"a", "b", "70m:real::1"}
    out = retest(sonnet, 0)
    assert out["units"] == 3 and out["real_units"] == 1 and out["same_label"] == 1.0
    versus = retest_versus_haiku({"haiku": haiku, "sonnet": sonnet}, "sonnet", 0)
    assert versus["units"] == 3 and versus["difference"] > 0


def test_agreement_is_kappa_on_real_neurons_per_model_and_overall():
    rows = rows_with(readers(SEPARATING, SEPARATING))
    out = agreement(rows, "haiku", "sonnet", 0)
    assert all(abs(k - 1) < 1e-9 for k in out["per_model"].values()) and abs(out["all_models"]["kappa"] - 1) < 1e-9
    assert out["all_models"]["n"] == 600


def test_label_shares_on_real_neurons():
    shares = label_shares(rows_with(readers(SEPARATING, WEAK)), "haiku")
    assert shares["all"] == {"A": 0.9, "B": 0.0, "C": 0.1} and np.isclose(sum(shares["70m"].values()), 1)
