import numpy as np

from recgen.analyze_v15 import analyze, unit_rows
from recgen.rubric_v8 import MODELS
from recgen.rubric_v15 import DETECTS, NO_DETECTION, RANDOM_STRATUM, TOP_STRATA

V8_AUC = {"minilm": 0.51, "network": 0.62, "haiku": 0.62, "haiku agent": 0.56, "sonnet agent": 0.64}


def reading(predicted: list[float] | None, explanation: str) -> dict:
    def record(parsed):
        return {"parsed": parsed, "attempts": [{"error": None}], "models": ["m"], "cost_usd": 0.01,
                "input_tokens": 10, "output_tokens": 5}
    return {"explanation": record(explanation), "simulations": {"sonnet": record(predicted), "haiku": record(predicted)}}


def fake_answers(pools_covered: bool) -> dict:
    """Per size: 4 real neurons, 4 50/50 pools and 2 related pools, 60 held-out examples each. Real neurons
    are predicted exactly. A pool's first neuron drives 24 of its 40 top-strata examples; unless
    pools_covered, the explanation only covers that neuron, so the other's examples are predicted 0."""
    rng = np.random.default_rng(0)
    units = {}
    for model in MODELS:
        for i, group in enumerate(["real"] * 4 + ["pooled 50/50"] * 4 + ["related"] * 2):
            strata = [TOP_STRATA[0]] * 20 + [TOP_STRATA[1]] * 20 + [RANDOM_STRATUM] * 20
            truth = list(rng.uniform(5, 8, 20)) + list(rng.uniform(3, 5, 20)) + list(rng.uniform(0, 2, 20))
            drivers = [0] * 12 + [1] * 8 + [0] * 12 + [1] * 8 + [0, 1] * 10 if group != "real" else [0] * 60
            predicted = [t if pools_covered or d == 0 or s == RANDOM_STRATUM else 0.0
                         for t, d, s in zip(truth, drivers, strata)]
            unit_id = f"{model}:{group.split()[0]}:{i}"
            units[unit_id] = {
                "model": model, "group": group, "neuron_ids": [i], "log_peak": [float(i)],
                "held_out": [{"value": t, "stratum": s, "driver": d} for t, s, d in zip(truth, strata, drivers)],
                "first": {**reading(predicted, "dates or times" if group != "real" else "dates"),
                          "judge": {"parsed": group != "real", "attempts": [{}], "models": ["j"], "cost_usd": 0.0,
                                    "input_tokens": 1, "output_tokens": 1}}}
            if i in (0, 4):
                units[unit_id]["retest"] = reading(predicted, "dates again")
    return {"units": units}


def test_unit_rows_score_each_reading_and_cover_pools_only():
    rows = unit_rows(fake_answers(pools_covered=False))
    real, pool = rows["70m"][0], rows["70m"][4]
    assert np.isclose(real["first"]["scores"]["sonnet"]["pearson"], 1) and real["first"]["coverage"]["sonnet"] is None
    assert pool["first"]["scores"]["sonnet"]["pearson"] < 0.9 and pool["retest"] is not None
    cover = pool["first"]["coverage"]["haiku"]
    assert cover["majority examples"] == 24 and cover["majority auc"] == 1 and cover["minority auc"] == 0
    assert real["words"] == 1 and not real["marker"] and pool["marker"] and rows["70m"][1]["retest"] is None


def test_a_score_that_misses_one_pool_neuron_detects_mixtures():
    s = analyze(fake_answers(pools_covered=False), V8_AUC)["summary"]
    assert s["verdict"] == DETECTS and s["primary"]["pooled 50/50 minus real"] < 0
    assert s["primary"]["auc, real over pooled 50/50"] == 1 and s["primary"]["auc interval above v8's largest label auc"]
    assert s["coverage"]["sonnet"]["majority minus minority"] == 1 and s["coverage"]["sonnet"]["pools"] == 12
    assert s["disjunction (exploratory)"]["pooled 50/50"]["share the judge calls unrelated"] == 1
    assert s["disjunction (exploratory)"]["real"]["share with a marker"] == 0
    assert s["retest"]["sonnet"]["units"] == 6 and s["retest"]["sonnet"]["mean absolute difference"] == 0
    assert s["descriptive"]["readings"]["judge"]["readings"] == 30 and s["descriptive"]["total cost_usd"] > 0


def test_a_score_that_covers_both_neurons_doesnt_detect_them():
    s = analyze(fake_answers(pools_covered=True), V8_AUC)["summary"]
    assert s["verdict"] == NO_DETECTION and abs(s["primary"]["pooled 50/50 minus real"]) < 1e-9
    assert s["coverage"]["sonnet"]["majority minus minority"] == 0


def test_units_without_valid_predictions_are_left_out_and_counted():
    answers = fake_answers(pools_covered=True)
    answers["units"]["70m:real:0"]["first"]["simulations"]["haiku"]["parsed"] = None
    s = analyze(answers, V8_AUC)["summary"]
    assert s["other simulator"]["haiku"]["units"] == s["primary"]["units"] - 1
    assert s["descriptive"]["readings"]["simulator haiku, first"]["without a valid reply"] == 1
