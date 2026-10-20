"""Apply the pre-registered v24 rules (recgen/rubric_v24.py): what Haiku, Sonnet and Opus, one shot under v7's
rubric and E3, say about 100 random-text units, beside 100 real neurons and 100 known 50/50 pools read in the
same run.

Reads results/v24_answers.json (from recgen.reader_v24) and, for the checks against earlier rounds of the same
prompts, v20's random-text readings (branch v20, through git), v7's Haiku answers, v10's Sonnet cache
(data/v10/sonnet/, read only), v16's answers and v21's answers. A unit's size and group come from its id. A unit
a series never labeled is left out of that series and, pairwise, out of paired comparisons.
"""

import json
from collections import Counter

import numpy as np

from recgen.analyze_v16 import one_main, patterns_of, rank, retest, rounded, share, strata, v10_sonnet
from recgen.llm_reader import ANSWERS_PATH as V7_ANSWERS_PATH
from recgen.llm_reader_v16 import ANSWERS_PATH as V16_ANSWERS_PATH
from recgen.reader_v21 import ANSWERS_PATH as V21_ANSWERS_PATH
from recgen.reader_v24 import ANSWERS_PATH, v20_random_answers
from recgen.rubric_v5 import MODELS
from recgen.rubric_v16 import auc, bootstrap
from recgen.rubric_v21 import separates
from recgen.rubric_v24 import (
    ARMS, C_LABEL, E3, GROUPS, ONE_JOB, POOL, RANDOM, READERS, REAL, RELATED, V7_RUBRIC, floor_verdict, position,
    position_reading,
)
from recgen.config import RESULTS_DIR

LABELS = (ONE_JOB, RELATED, C_LABEL)
SHARES_FOR_POSITION = {"one (main) job (A)": ONE_JOB, "C (unrelated jobs, or not yet understood)": C_LABEL}
AUCS = {"random over real": (RANDOM, REAL), "random over pools": (RANDOM, POOL), "pools over real": (POOL, REAL)}
EARLIER = {"haiku-v7": "v7", "sonnet-v7": "v10", "haiku-E3": "v16", "sonnet-E3": "v16", "opus-v7": "v21", "opus-E3": "v21"}
V20_HAIKU = "v20:haiku-v7"  # v20's readings of the random-text units, same prompts, another round
ESSAY_ONE_MAIN_JOB = 0.90
EXAMPLES = 3


def unit_place(unit_id: str) -> tuple[str, str]:
    """A v24 unit's size and group, from its id (size:kind:mix:...)."""
    size, kind, mix = unit_id.split(":")[:3]
    return size, {"random": RANDOM, "real": REAL}.get(kind, f"pooled {mix}")


def is_label(label: str):
    return lambda _series, x: float(x == label)


def unit_rows(answers: dict, earlier: dict) -> list[dict]:
    """Every v24 unit with its first-reading label in each v24 series and in each earlier round (None if missing)."""
    ids = sorted({k for parts in answers["series"].values() for k in parts["first"]})
    rows = []
    for unit_id in ids:
        size, group = unit_place(unit_id)
        labels = {s: (answers["series"][s]["first"].get(unit_id) or {}).get("label") for s in ARMS}
        labels |= {name: labels_of.get(unit_id) for name, labels_of in earlier.items()}
        rows.append({"id": unit_id, "size": size, "group": group, "labels": labels})
    return rows


def auc_between(st: dict, upper: str, lower: str, column: int = 0) -> float:
    """Mean over sizes of AUC(upper over lower) on label ranks."""
    sizes = sorted({k[0] for k in st}, key=MODELS.index)
    return float(np.mean([auc(st[(s, upper)][:, column], st[(s, lower)][:, column]) for s in sizes]))


def gap(st: dict, a: str, b: str) -> float:
    return share(st, a) - share(st, b)


def position_of(st: dict) -> float:
    denominator = gap(st, POOL, RANDOM)
    return float("nan") if denominator == 0 else position(share(st, REAL), share(st, RANDOM), share(st, POOL))


def series_stats(rows: list[dict], series: str, rng) -> dict:
    """P1 to P6 and S2 for one series."""
    out = {"labels": {}, "shares": {}, "gaps": {}, "position": {}, "label_auc": {}}
    for g in GROUPS:
        out["labels"][g] = dict(sorted(Counter(u["labels"][series] for u in rows if u["group"] == g).items(),
                                       key=lambda kv: str(kv[0])))
        out["shares"][g] = {lab: bootstrap(strata(rows, [series], [g], is_label(lab)), lambda s, g=g: share(s, g), rng)
                            for lab in LABELS}
        out["shares"][g]["units"] = sum(u["labels"][series] is not None for u in rows if u["group"] == g)
    floor = out["shares"][RANDOM][ONE_JOB]
    out["floor"] = {**floor, "verdict": floor_verdict(floor["interval"])}
    for name, lab in SHARES_FOR_POSITION.items():
        st = strata(rows, [series], GROUPS, is_label(lab))
        real_gap = bootstrap(st, lambda s: gap(s, REAL, RANDOM), rng)
        pool_gap = bootstrap(st, lambda s: gap(s, POOL, RANDOM), rng)
        out["gaps"][name] = {"real minus random": real_gap, "pools minus random": pool_gap}
        defined = not (pool_gap["interval"][0] <= 0 <= pool_gap["interval"][1])
        p = bootstrap(st, position_of, rng) if defined else None
        out["position"][name] = {"p": p, "reading": position_reading(pool_gap["interval"], p and p["interval"])}
    ranks = strata(rows, [series], GROUPS, rank)
    for name, (upper, lower) in AUCS.items():
        value = bootstrap(ranks, lambda s, a=upper, b=lower: auc_between(s, a, b), rng)
        out["label_auc"][name] = {**value, "separates": separates(value["interval"])}
    one = strata(rows, [series], GROUPS, one_main)
    out["floor_corrected_real_one_job"] = bootstrap(
        one, lambda s: (share(s, REAL) - share(s, RANDOM)) / (1 - share(s, RANDOM)), rng)
    return out


def e3_minus_v7(rows: list[dict], reader: str, rng) -> dict:
    """S1: per group, the paired rise in the one-(main-)job share, E3 minus v7's rubric, on the same units."""
    st = strata(rows, [f"{reader}-{E3}", f"{reader}-{V7_RUBRIC}"], GROUPS, one_main)
    return {g: bootstrap(st, lambda s, g=g: share(s, g, 0) - share(s, g, 1), rng) for g in GROUPS}


def against_earlier(rows: list[dict], series: str, earlier: str, groups, rng) -> dict:
    """S4 and S5: this round against an earlier round of the same prompts, paired on units both labeled."""
    pairs = [(u["labels"][series], u["labels"][earlier]) for u in rows if u["group"] in groups
             and u["labels"][series] is not None and u["labels"].get(earlier) is not None]
    out = {"units": len(pairs), "same label": float(np.mean([a == b for a, b in pairs])) if pairs else None}
    if not pairs:
        return out
    for name, lab in (("A share, now minus earlier", ONE_JOB), ("C share, now minus earlier", C_LABEL)):
        st = strata(rows, [series, earlier], groups, is_label(lab))
        out[name] = {g: bootstrap(st, lambda s, g=g: share(s, g, 0) - share(s, g, 1), rng) for g in groups}
    return out


def not_one(label: str) -> bool:
    return label != ONE_JOB


def retests(answers: dict, rng) -> dict:
    """S3: two readings of the 30 retest units per series, with the "not one (main) job" flag."""
    out = {}
    for s, parts in answers["series"].items():
        out[s] = retest({k: a["label"] for k, a in parts["first"].items()},
                        {k: a["label"] for k, a in parts["retest"].items()}, not_one, rng)
    return out


def usage(answers: dict) -> dict:
    """S6: CLI-reported cost, tokens and resolved model ids per series, the pilot, and the spend log's total."""
    out = {}
    for s, parts in answers["series"].items():
        done = list(parts["first"].values()) + list(parts["retest"].values())
        out[s] = {"first readings": len(parts["first"]), "retests": len(parts["retest"]),
                  "without a label": sum(a["label"] is None for a in done),
                  "models": sorted({m for a in done for m in a["models"]}),
                  "cost_usd": sum(a["cost_usd"] or 0 for a in done),
                  "input_tokens": sum(a["input_tokens"] for a in done),
                  "mean output tokens": float(np.mean([a["output_tokens"] for a in done])) if done else None,
                  "mean thinking tokens": float(np.mean([a["thinking_tokens"] for a in done])) if done else None}
    pilot = [a for part in answers["pilot"].values() for a in part.values()]
    out["pilot"] = {"readings": len(pilot), "labeled": sum(a["label"] is not None for a in pilot),
                    "cost_usd": sum(a["cost_usd"] or 0 for a in pilot)}
    out["all v24 CLI calls, by the spend log"] = answers["spend_usd"]
    return out


def exploratory(rows: list[dict], answers: dict) -> dict:
    """Listed in the rubric: patterns per one-(main-)job answer by group, and examples of random text called A."""
    out = {"one_job_answers": {}, "random_text_called_one_job": {}}
    for s in ARMS:
        patterns = {k: patterns_of(a["reply"]) for k, a in answers["series"][s]["first"].items()}
        out["one_job_answers"][s] = {}
        for g in GROUPS:
            lists = [patterns[u["id"]] for u in rows if u["group"] == g and u["labels"][s] == ONE_JOB]
            out["one_job_answers"][s][g] = {
                "answers": len(lists), "mean patterns": float(np.mean([len(p) for p in lists])) if lists else None,
                "share listing 2 or more": float(np.mean([len(p) >= 2 for p in lists])) if lists else None}
        chosen = [u for u in rows if u["group"] == RANDOM and u["labels"][s] == ONE_JOB]
        out["random_text_called_one_job"][s] = [{"id": u["id"], "patterns": patterns[u["id"]]}
                                                for u in chosen[:: max(1, len(chosen) // EXAMPLES)][:EXAMPLES]]
    return out


def earlier_rounds(v20: dict, v7: dict, sonnet: dict | None, v16: dict, v21: dict) -> dict:
    """Labels from earlier rounds of the same prompts, unit id -> label, per earlier series."""
    out = {V20_HAIKU: {k: r["label"] for k, r in v20.items()},
           "v7:haiku-v7": {k: a["label"] for k, a in v7["first"].items()},
           "v16:haiku-E3": {k: a["label"] for k, a in v16["arms"]["haiku-E3"]["first"].items()},
           "v16:sonnet-E3": {k: a["label"] for k, a in v16["arms"]["sonnet-E3"]["first"].items()},
           "v21:opus-v7": {k: a["label"] for k, a in v21["arm_a"]["opus-v7"]["first"].items()},
           "v21:opus-E3": {k: a["label"] for k, a in v21["arm_a"]["opus-E3"]["first"].items()}}
    if sonnet is not None:
        out["v10:sonnet-v7"] = sonnet["first"]
    return out


def prompt_checks(answers: dict, v20: dict, v7: dict, v16: dict, v21: dict) -> dict:
    """Each v24 prompt against the earlier round's prompt for the same unit, by hash (v10's cache was checked
    against v7's hashes by analyze_v16.v10_sonnet)."""
    references = {"haiku-v7": v7["first"] | v20, "sonnet-v7": v7["first"], "opus-v7": v21["arm_a"]["opus-v7"]["first"],
                  "haiku-E3": v16["arms"]["haiku-E3"]["first"], "sonnet-E3": v16["arms"]["sonnet-E3"]["first"],
                  "opus-E3": v21["arm_a"]["opus-E3"]["first"]}
    out = {}
    for s, reference in references.items():
        first = answers["series"][s]["first"]
        out[s] = {"readings with an earlier reading": sum(k in reference for k in first),
                  "same prompt": sum(a["prompt_sha256"] == reference[k]["prompt_sha256"] for k, a in first.items()
                                     if k in reference)}
    out["random-text prompts rebuilt and checked against v20's before reading"] = answers["random_prompts_matching_v20"]
    return out


def analyze(answers: dict, v20: dict, v7: dict, sonnet: dict | None, v16: dict, v21: dict, rng) -> dict:
    earlier = earlier_rounds(v20, v7, sonnet, v16, v21)
    rows = unit_rows(answers, earlier)
    summary = {
        "series": {s: series_stats(rows, s, rng) for s in ARMS},
        "e3_minus_v7": {r: e3_minus_v7(rows, r, rng) for r in READERS},
        "haiku_v7_random_against_v20": against_earlier(rows, "haiku-v7", V20_HAIKU, (RANDOM,), rng),
        "anchor_against_earlier_rounds": {
            s: {"earlier": f"{EARLIER[s]}:{s}", **against_earlier(rows, s, f"{EARLIER[s]}:{s}", (REAL, POOL), rng)}
            for s in ARMS if f"{EARLIER[s]}:{s}" in earlier},
        "retest": retests(answers, rng),
        "essay_one_main_job": ESSAY_ONE_MAIN_JOB,
    }
    return {"summary": rounded(summary), "exploratory": rounded(exploratory(rows, answers)),
            "usage": rounded(usage(answers)), "checks": prompt_checks(answers, v20, v7, v16, v21), "units": rows}


def main() -> None:
    answers = json.loads(ANSWERS_PATH.read_text())
    v7 = json.loads(V7_ANSWERS_PATH.read_text())
    v16 = json.loads(V16_ANSWERS_PATH.read_text())
    v21 = json.loads(V21_ANSWERS_PATH.read_text())
    result = analyze(answers, v20_random_answers(), v7, v10_sonnet(v7), v16, v21, np.random.default_rng([24, 1]))
    (RESULTS_DIR / "classification_v24.json").write_text(json.dumps(result, indent=1))
    for s, st in result["summary"]["series"].items():
        sh = st["shares"]
        print(f"{s}: A random {sh[RANDOM][ONE_JOB]['value']:.2f} {sh[RANDOM][ONE_JOB]['interval']} ({st['floor']['verdict']}), "
              f"real {sh[REAL][ONE_JOB]['value']:.2f}, pools {sh[POOL][ONE_JOB]['value']:.2f}; C random "
              f"{sh[RANDOM][C_LABEL]['value']:.2f}, real {sh[REAL][C_LABEL]['value']:.2f}, pools {sh[POOL][C_LABEL]['value']:.2f}")
        for name, pos in st["position"].items():
            print(f"   position on {name}: {pos['p'] and pos['p']['value']} {pos['p'] and pos['p']['interval']}: {pos['reading']}")
        print("   AUC " + ", ".join(f"{k} {v['value']:.2f} {v['interval']}" for k, v in st["label_auc"].items()))
    print(json.dumps(result["usage"], indent=1))
    print(json.dumps(result["checks"], indent=1))


if __name__ == "__main__":
    main()
