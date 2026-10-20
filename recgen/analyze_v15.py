"""Apply the pre-registered v15 rules (recgen/rubric_v15.py): does the standard simulation score tell
v8's 50/50 pools from real neurons, and how does its power compare with v8's label readers?

Reads results files only: v15's readings (results/v15_answers.json, from recgen.simulate_v15) and
v8's label AUCs (results/classification_v8.json). A unit a simulator never scored is left out of
that simulator's numbers and counted.
"""

import json

import numpy as np
from scipy.stats import spearmanr

from recgen.config import RESULTS_DIR
from recgen.rubric_v8 import BOOTSTRAP, MODELS
from recgen.rubric_v15 import (
    MIN_DRIVEN, PRIMARY_SIMULATOR, SCREEN_AUC, SIMULATORS, UNCOVERED_AUC, coverage, group_difference,
    has_disjunction_marker, pearson, separation_auc, unit_scores, verdict,
)
from recgen.simulate_v15 import ANSWERS_PATH, READINGS

ARMS = tuple(SIMULATORS)
GROUPS = ("real", "pooled 50/50", "related")
SCORES = ("pearson", "spearman", "detection_auc")
V8_READERS = ("minilm", "network", "haiku", "haiku agent", "sonnet agent")


def predictions(part: dict | None, arm: str) -> list[float] | None:
    simulation = (part or {}).get("simulations", {}).get(arm)
    return None if simulation is None else simulation["parsed"]


def unit_rows(answers: dict) -> dict[str, list[dict]]:
    """Each unit of each size with its explanations, its scores by simulator and reading (None where a
    simulator gave no valid predictions) and, for pools, each neuron's coverage."""
    rows = {model: [] for model in MODELS}
    for unit_id, u in answers["units"].items():
        truth = [h["value"] for h in u["held_out"]]
        strata = [h["stratum"] for h in u["held_out"]]
        drivers = [h["driver"] for h in u["held_out"]]
        row = {"id": unit_id, "model": u["model"], "group": u["group"], "neuron_ids": u["neuron_ids"],
               "log_peak": u["log_peak"], "judge": (u["first"].get("judge") or {}).get("parsed")}
        for reading in READINGS:
            part = u.get(reading)
            if part is None:
                row[reading] = None
                continue
            explanation = (part["explanation"] or {}).get("parsed")
            preds = {arm: predictions(part, arm) for arm in ARMS}
            row[reading] = {
                "explanation": explanation,
                "scores": {arm: None if p is None else unit_scores(p, truth, strata) for arm, p in preds.items()},
                "coverage": {arm: coverage(p, strata, drivers) if p is not None and u["group"] != "real" else None
                             for arm, p in preds.items()}}
        explanation = row["first"]["explanation"]
        row["words"] = None if explanation is None else len(explanation.split())
        row["marker"] = None if explanation is None else has_disjunction_marker(explanation)
        rows[u["model"]].append(row)
    return rows


def score(row: dict, arm: str, name: str, reading: str = "first") -> float | None:
    part = row[reading]
    return None if part is None or part["scores"][arm] is None else part["scores"][arm][name]


def per_model(rows: dict, arm: str, name: str, groups: tuple[str, str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per size, the two groups' scores from one simulator, on units it scored."""
    return [tuple(np.array([s for u in rows[m] if u["group"] == g and (s := score(u, arm, name)) is not None])
                  for g in groups) for m in MODELS]


def compare(rows: dict, arm: str, name: str, rng, lower: str = "pooled 50/50", higher: str = "real") -> dict:
    """lower minus higher (mean over sizes, with an interval) and the AUC of higher over lower."""
    pairs = per_model(rows, arm, name, (lower, higher))
    point, interval = group_difference(pairs, rng)
    auc_point, auc_interval = separation_auc([(b, a) for a, b in pairs], rng)
    return {f"{lower} minus {higher}": round(point, 3), "interval": [round(x, 3) for x in interval],
            "per_model": {m: round(float(a.mean() - b.mean()), 3) for m, (a, b) in zip(MODELS, pairs)},
            f"auc, {higher} over {lower}": round(auc_point, 3), "auc_interval": [round(x, 3) for x in auc_interval],
            "units": int(sum(len(a) + len(b) for a, b in pairs))}


def mixture_test(rows: dict, arm: str, rng, v8_auc: dict) -> dict:
    """The pre-registered test for one simulator, with its verdict and the descriptive comparisons."""
    out = compare(rows, arm, "pearson", rng)
    out["verdict"] = verdict(out["interval"], out["auc_interval"])
    best = max(v8_auc.values())
    out["auc interval above v8's largest label auc"] = out["auc_interval"][0] > best
    out[f"auc at least {SCREEN_AUC}"] = out["auc, real over pooled 50/50"] >= SCREEN_AUC
    return out


def paired_mean(per_model: list[np.ndarray], rng) -> tuple[float, list[float]]:
    """Mean over sizes of a per-unit difference, with a bootstrap interval resampling units within size."""
    draws = [np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in per_model]) for _ in range(BOOTSTRAP)]
    return float(np.mean([d.mean() for d in per_model])), [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def coverage_stats(rows: dict, arm: str, rng, group: str = "pooled 50/50") -> dict:
    """How well each pool's majority and minority neuron are predicted (rubric secondary 4)."""
    pools = [[u for u in rows[m] if u["group"] == group and u["first"]["coverage"][arm] is not None] for m in MODELS]
    both = [[u["first"]["coverage"][arm] for u in p if u["first"]["coverage"][arm]["minority auc"] is not None] for p in pools]
    point, interval = paired_mean([np.array([c["majority auc"] - c["minority auc"] for c in b]) for b in both], rng)
    flat = [c for b in both for c in b]
    real = [s for m in MODELS for u in rows[m] if u["group"] == "real" and (s := score(u, arm, "detection_auc")) is not None]
    return {"pools": sum(map(len, pools)), f"pools whose minority neuron drives fewer than {MIN_DRIVEN}": sum(map(len, pools)) - len(flat),
            "mean majority auc": round(float(np.mean([c["majority auc"] for c in flat])), 3),
            "mean minority auc": round(float(np.mean([c["minority auc"] for c in flat])), 3),
            "majority minus minority": round(point, 3), "interval": [round(x, 3) for x in interval],
            "median minority share of top-strata examples": round(float(np.median(
                [c["minority examples"] / (c["minority examples"] + c["majority examples"]) for b in pools for c in
                 (u["first"]["coverage"][arm] for u in b)])), 3),
            f"share of pools with minority auc below {UNCOVERED_AUC}": round(float(np.mean(
                [c["minority auc"] < UNCOVERED_AUC for c in flat])), 3),
            f"share of pools with majority auc below {UNCOVERED_AUC}": round(float(np.mean(
                [c["majority auc"] < UNCOVERED_AUC for c in flat])), 3),
            f"reference: share of real neurons with detection auc below {UNCOVERED_AUC}": round(float(np.mean(
                [s < UNCOVERED_AUC for s in real])), 3)}


def disjunction_stats(rows: dict) -> dict:
    """Exploratory: marker matches, the judge's yes and word counts by unit type, and pools' scores
    with and without the judge's yes."""
    out = {}
    for g in GROUPS:
        units = [u for m in MODELS for u in rows[m] if u["group"] == g and u["first"]["explanation"] is not None]
        judged = [u for u in units if u["judge"] is not None]
        out[g] = {"explanations": len(units), "share with a marker": round(float(np.mean([u["marker"] for u in units])), 3),
                  "share the judge calls unrelated": round(float(np.mean([u["judge"] for u in judged])), 3),
                  "judged": len(judged), "mean words": round(float(np.mean([u["words"] for u in units])), 1),
                  "words, 10th to 90th percentile": [int(x) for x in np.percentile([u["words"] for u in units], [10, 90])]}
        for arm in ARMS:
            for verdict_value, name in ((True, "judge yes"), (False, "judge no")):
                scores = [s for u in judged if u["judge"] is verdict_value and (s := score(u, arm, "pearson")) is not None]
                out[g][f"{arm} mean score, {name}"] = [round(float(np.mean(scores)), 3) if scores else None, len(scores)]
    return out


def retest_stats(rows: dict, rng) -> dict:
    out = {}
    for arm in ARMS:
        pairs = np.array([(score(u, arm, "pearson"), score(u, arm, "pearson", "retest")) for m in MODELS for u in rows[m]
                          if u["retest"] is not None and score(u, arm, "pearson") is not None
                          and score(u, arm, "pearson", "retest") is not None])
        draws = [pearson(*pairs[idx].T) for idx in (rng.integers(0, len(pairs), len(pairs)) for _ in range(BOOTSTRAP))]
        out[arm] = {"units": len(pairs), "correlation of the two runs' scores": round(pearson(*pairs.T), 3),
                    "interval": [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])],
                    "mean absolute difference": round(float(np.mean(np.abs(pairs[:, 0] - pairs[:, 1]))), 3),
                    "mean first, mean retest": [round(float(x), 3) for x in pairs.mean(axis=0)]}
    return out


def peak_stats(rows: dict, rng, arm: str = PRIMARY_SIMULATOR) -> dict:
    """Exploratory: real neurons' simulation score against v12's log peak."""
    def pairs_of(models):
        return np.array([(u["log_peak"][0], s) for m in models for u in rows[m] if u["group"] == "real"
                         and (s := score(u, arm, "pearson")) is not None])
    everything = pairs_of(MODELS)
    rho = lambda p: float(spearmanr(p[:, 0], p[:, 1]).statistic)  # noqa: E731
    draws = [rho(everything[rng.integers(0, len(everything), len(everything))]) for _ in range(BOOTSTRAP)]
    return {"spearman, all sizes": round(rho(everything), 3), "interval": [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])],
            "per_model": {m: round(rho(pairs_of([m])), 3) for m in MODELS}, "units": len(everything)}


def descriptive(rows: dict, answers: dict) -> dict:
    means = {m: {arm: {g: {name: round(float(np.mean(v)), 3) if (v := [s for u in rows[m] if u["group"] == g
                                                                        and (s := score(u, arm, name)) is not None]) else None
                          for name in SCORES} for g in GROUPS} for arm in ARMS} for m in MODELS}
    both = np.array([(score(u, "sonnet", "pearson"), score(u, "haiku", "pearson")) for m in MODELS for u in rows[m]
                     if score(u, "sonnet", "pearson") is not None and score(u, "haiku", "pearson") is not None])
    records = {}
    for u in answers["units"].values():
        for reading in READINGS:
            part = u.get(reading)
            if part is None:
                continue
            records.setdefault(f"explainer, {reading}", []).append(part["explanation"])
            for arm in ARMS:
                records.setdefault(f"simulator {arm}, {reading}", []).append(part["simulations"][arm])
        records.setdefault("judge", []).append(u["first"].get("judge"))
    readings = {role: {"readings": sum(r is not None for r in rs), "without a valid reply": sum(r is None or r["parsed"] is None for r in rs),
                       "with retries": sum(r is not None and len(r["attempts"]) > 1 for r in rs),
                       "models used": sorted({m for r in rs if r for m in r["models"]}),
                       "cost_usd": round(sum(r["cost_usd"] for r in rs if r), 2),
                       "input_tokens": sum(r["input_tokens"] for r in rs if r), "output_tokens": sum(r["output_tokens"] for r in rs if r)}
                for role, rs in records.items()}
    constant = {arm: sum(1 for m in MODELS for u in rows[m] if u["first"]["scores"][arm] and u["first"]["scores"][arm]["constant"])
                for arm in ARMS}
    return {"mean scores": means, "simulators agree, pearson of their scores": round(pearson(*both.T), 3),
            "constant predictions": constant, "readings": readings,
            "total cost_usd": round(sum(r["cost_usd"] for r in readings.values()), 2)}


def analyze(answers: dict, v8_auc: dict) -> dict:
    rows = unit_rows(answers)
    rng = np.random.default_rng([15, 99])
    stats = {"primary": mixture_test(rows, PRIMARY_SIMULATOR, rng, v8_auc),
             "other simulator": {arm: mixture_test(rows, arm, rng, v8_auc) for arm in ARMS if arm != PRIMARY_SIMULATOR},
             "v8 label aucs, pools over real": v8_auc,
             "related": {arm: {"related vs real": compare(rows, arm, "pearson", rng, "related", "real"),
                               "pooled vs related": compare(rows, arm, "pearson", rng, "pooled 50/50", "related")}
                         for arm in ARMS},
             "other scores": {arm: {name: compare(rows, arm, name, rng) for name in ("spearman", "detection_auc")} for arm in ARMS},
             "coverage": {arm: coverage_stats(rows, arm, rng) for arm in ARMS},
             "coverage, related pools": {arm: coverage_stats(rows, arm, rng, "related") for arm in ARMS},
             "disjunction (exploratory)": disjunction_stats(rows),
             "retest": retest_stats(rows, rng),
             "peak (exploratory)": peak_stats(rows, rng),
             "descriptive": descriptive(rows, answers)}
    stats["verdict"] = stats["primary"]["verdict"]
    return {"summary": stats, "units": rows}


def main() -> None:
    answers = json.loads(ANSWERS_PATH.read_text())
    v8 = json.loads((RESULTS_DIR / "classification_v8.json").read_text())["summary"]["post_hoc"]["label_auc"]
    out = analyze(answers, {r: v8[r]["mean_over_models"] for r in V8_READERS})
    (RESULTS_DIR / "classification_v15.json").write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(json.dumps({k: s[k] for k in ("verdict", "primary", "other simulator", "coverage", "retest")}, indent=1))


if __name__ == "__main__":
    main()
