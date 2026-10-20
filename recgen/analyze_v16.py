"""Apply the pre-registered v16 rules (recgen/rubric_v16.py): the essay's rubrics against v7's.

Reads results/v16_answers.json (from recgen.llm_reader_v16), v7's Haiku answers
(results/v7_llm_answers.json) and v10's Sonnet answers (data/v10/sonnet/, read only). A unit's size
and group come from its id, so no recording is loaded. A unit an arm never labeled is left out of
that arm's rates and, pairwise, out of paired comparisons.
"""

import json
from collections import Counter

import numpy as np

from recgen.config import DATA_DIR, RESULTS_DIR
from recgen.llm_reader import ANSWERS_PATH as V7_ANSWERS_PATH
from recgen.llm_reader import cache_path
from recgen.llm_reader_v16 import ANSWERS_PATH
from recgen.rubric_v5 import MODELS
from recgen.rubric_v6 import cohen_kappa
from recgen.rubric_v16 import (
    ARMS, BOOTSTRAP, ESSAY_ONE_MAIN_JOB, LABEL_RANK, LABELS, ONE_MAIN_JOB, READERS, RUBRICS, UNRELATED_JOBS,
    V7_RUBRIC, V7_UNRELATED, auc, bootstrap, changes, interval, near_essay, verdict,
)

V10_SONNET_DIR = DATA_DIR / "v10" / "sonnet"
POOL = "pooled 50/50"
GROUPS = ("real", POOL, "pooled 70/30", "pooled 90/10", "related")
ESSAY_SHARES = {"one main job": 0.90, "two related jobs": 0.07, "not yet understood": "0.02 to 0.04"}
EXAMPLES = 3  # per arm, 50/50 pools called one main job, for the README


def rubric_of(series: str) -> str:
    return series.split("-")[1]


def unit_place(unit_id: str) -> tuple[str, str]:
    """A v7 unit's size and group, from its id (size:kind:mix:neurons)."""
    size, kind, mix = unit_id.split(":")[:3]
    return size, "real" if kind == "real" else "related" if kind == "related" else f"pooled {mix}"


def v10_sonnet(v7_answers: dict) -> dict | None:
    """v10's Sonnet readings under v7's rubric ({"first", "retest"}: unit id -> label), if every one
    of v7's units has a cached first reading of v7's exact prompt; otherwise None."""
    out = {"first": {}, "retest": {}, "models": set()}
    for part, folder in (("first", "llm"), ("retest", "llm_retest")):
        for unit_id, v7 in v7_answers[part].items():
            path = cache_path(V10_SONNET_DIR / folder, unit_id)
            record = json.loads(path.read_text()) if path.exists() else None
            if record is None or record["prompt_sha256"] != v7["prompt_sha256"]:
                if part == "first":
                    return None
                continue
            out[part][unit_id] = record["label"]
            out["models"] |= set(record["models"])
    out["models"] = sorted(out["models"])
    return out


def baselines(v7_answers: dict, sonnet: dict | None) -> dict:
    """Each reader's v7-rubric readings, as {"first", "retest"}: unit id -> label."""
    out = {"haiku-v7": {part: {k: a["label"] for k, a in v7_answers[part].items()} for part in ("first", "retest")}}
    if sonnet is not None:
        out["sonnet-v7"] = {part: sonnet[part] for part in ("first", "retest")}
    return out


def unit_rows(answers: dict, base: dict) -> list[dict]:
    """Every v7 unit with its label in each v16 arm and each v7-rubric baseline (None if missing)."""
    rows = []
    for unit_id in base["haiku-v7"]["first"]:
        size, group = unit_place(unit_id)
        labels = {arm: (answers["arms"][arm]["first"].get(unit_id) or {}).get("label") for arm in ARMS}
        labels |= {name: b["first"].get(unit_id) for name, b in base.items()}
        rows.append({"id": unit_id, "size": size, "group": group, "labels": labels})
    return rows


def one_main(series: str, label: str) -> float:
    return float(label == ONE_MAIN_JOB)


def not_one_main(series: str, label: str) -> float:
    return float(label != ONE_MAIN_JOB)


def unrelated(series: str, label: str) -> float:
    """The "unrelated jobs" flag: D in E4, C in v7's rubric."""
    return float(label == (V7_UNRELATED if rubric_of(series) == V7_RUBRIC else UNRELATED_JOBS))


def rank(series: str, label: str) -> float:
    return float(LABEL_RANK[rubric_of(series)][label])


def strata(rows: list[dict], series: list[str], groups, value, sizes=MODELS) -> dict:
    """(size, group) -> [units, series] values, on units labeled in every series."""
    out = {}
    for size in sizes:
        for group in groups:
            units = [u for u in rows if u["size"] == size and u["group"] == group
                     and all(u["labels"].get(s) is not None for s in series)]
            out[(size, group)] = np.array([[value(s, u["labels"][s]) for s in series] for u in units],
                                          float).reshape(len(units), len(series))
    return out


def sizes_of(st: dict) -> list[str]:
    return sorted({k[0] for k in st}, key=MODELS.index)


def share(st: dict, group: str, column: int = 0) -> float:
    """The mean over sizes of a column's mean in one group."""
    return float(np.mean([st[(s, group)][:, column].mean() for s in sizes_of(st)]))


def excess(st: dict, column: int = 0) -> float:
    """The mean over sizes of the 50/50 pools' mean minus the real neurons'."""
    return float(np.mean([st[(s, POOL)][:, column].mean() - st[(s, "real")][:, column].mean() for s in sizes_of(st)]))


def auc_over_sizes(st: dict, column: int = 0) -> float:
    return float(np.mean([auc(st[(s, POOL)][:, column], st[(s, "real")][:, column]) for s in sizes_of(st)]))


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    return round(x, 3) if isinstance(x, float) else x


def arm_stats(rows: list[dict], series: str, rng) -> dict:
    """Tests 1 to 3 for one arm (or one v7-rubric baseline): one-main-job shares by group, over sizes
    and per size; detection; the label AUC; label counts."""
    out = {"one_main_job": {}, "detection": {}, "labels": {}}
    for group in GROUPS:
        st = strata(rows, [series], [group], one_main)
        out["labels"][group] = dict(sorted(Counter(u["labels"][series] for u in rows if u["group"] == group).items(),
                                           key=lambda kv: str(kv[0])))
        if not all(len(v) for v in st.values()):
            out["one_main_job"][group] = {"value": None, "units": int(sum(len(v) for v in st.values()))}
            continue
        out["one_main_job"][group] = {
            **bootstrap(st, lambda s, g=group: share(s, g), rng),
            "per_size": {size: bootstrap({k: v for k, v in st.items() if k[0] == size}, lambda s, g=group: share(s, g), rng)
                         for size in MODELS},
            "units": int(sum(len(v) for v in st.values()))}
    out["near_essay"] = near_essay(out["one_main_job"]["real"]["interval"])
    flags = {"not one main job": not_one_main}
    if rubric_of(series) != "E3":
        flags["unrelated jobs"] = unrelated
    for name, value in flags.items():
        st = strata(rows, [series], [POOL, "real"], value)
        out["detection"][name] = {**bootstrap(st, excess, rng),
                                  "per_size": {s: float(st[(s, POOL)][:, 0].mean() - st[(s, "real")][:, 0].mean())
                                               for s in MODELS}}
    out["label_auc"] = bootstrap(strata(rows, [series], [POOL, "real"], rank), auc_over_sizes, rng)
    return out


def versus_v7(rows: list[dict], reader: str, rubric: str, rng) -> dict:
    """The pre-registered verdict: rises in the one-main-job share of real neurons and of 50/50
    pools, this rubric minus v7's, paired on the same units."""
    st = strata(rows, [f"{reader}-{rubric}", f"{reader}-{V7_RUBRIC}"], ["real", POOL], one_main)

    def rise(s, group):
        return share(s, group, 0) - share(s, group, 1)

    rise_real = bootstrap(st, lambda s: rise(s, "real"), rng)
    rise_pool = bootstrap(st, lambda s: rise(s, POOL), rng)
    diff = bootstrap(st, lambda s: rise(s, POOL) - rise(s, "real"), rng)
    return {"rise_real": rise_real, "rise_pool": rise_pool, "rise_pool_minus_rise_real": diff,
            "verdict": verdict(rise_real["interval"], diff["interval"]),
            "units": int(sum(len(v) for v in st.values()))}


def e4_versus_e3(rows: list[dict], reader: str, rng) -> dict:
    """Question 3, paired: E4 minus E3 in one-main-job shares and detection, and E4's "unrelated
    jobs" detection minus the v7 rubric's."""
    e4, e3, v7 = (f"{reader}-{r}" for r in ("E4", "E3", V7_RUBRIC))
    st = strata(rows, [e4, e3], ["real", POOL], one_main)
    flags = strata(rows, [e4, e3], [POOL, "real"], not_one_main)
    out = {"real one main job": bootstrap(st, lambda s: share(s, "real", 0) - share(s, "real", 1), rng),
           "50/50 one main job": bootstrap(st, lambda s: share(s, POOL, 0) - share(s, POOL, 1), rng),
           "detection, not one main job": bootstrap(flags, lambda s: excess(s, 0) - excess(s, 1), rng)}
    if any(r["labels"].get(v7) is not None for r in rows):
        both = strata(rows, [e4, v7], [POOL, "real"], unrelated)
        out["detection, unrelated jobs, E4 minus v7 rubric"] = bootstrap(both, lambda s: excess(s, 0) - excess(s, 1), rng)
    return {k: {**v, "changes": changes(v["interval"])} for k, v in out.items()}


def retest(first: dict, second: dict, flag, rng) -> dict:
    """Test 4: agreement of two readings of the retest units, with a bootstrap interval of the
    flag's kappa (resampling the units)."""
    pairs = [(first.get(k), v, k) for k, v in second.items()]
    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
    a, b = (np.array([flag(p[i]) for p in pairs], bool) for i in (0, 1))
    draws = [cohen_kappa(a[idx], b[idx]) for idx in (rng.integers(0, len(pairs), len(pairs)) for _ in range(BOOTSTRAP))]
    return {"units": len(pairs), "real units": sum(":real:" in k for *_, k in pairs),
            "same_label": float(np.mean([x == y for x, y, _ in pairs])), "same_flag": float(np.mean(a == b)),
            "flag_kappa": cohen_kappa(a, b), "flag_kappa_interval": interval(draws)}


def patterns_of(reply: str) -> list[str]:
    try:
        start, end = reply.index("{"), reply.rindex("}") + 1
        patterns = json.loads(reply[start:end]).get("patterns") or []
    except (ValueError, AttributeError):
        return []
    return [str(p) for p in patterns] if isinstance(patterns, list) else []


def exploratory(rows: list[dict], answers: dict, base: dict) -> dict:
    """Not pre-registered: where each unit's v7-rubric label went, how many patterns one-main-job
    answers list, the essay's label shares beside E3's, and examples."""
    out = {"label_moves": {}, "patterns_per_one_main_job_answer": {}, "essay_shares_vs_real_labels": {},
           "examples": {}}
    for reader in READERS:
        if f"{reader}-{V7_RUBRIC}" not in base:
            continue
        for rubric in RUBRICS:
            moves = {}
            for group in ("real", POOL):
                pairs = Counter(f"{u['labels'][f'{reader}-{V7_RUBRIC}']} -> {u['labels'][f'{reader}-{rubric}']}"
                                for u in rows if u["group"] == group and u["labels"][f"{reader}-{rubric}"] is not None
                                and u["labels"][f"{reader}-{V7_RUBRIC}"] is not None)
                moves[group] = dict(sorted(pairs.items()))
            out["label_moves"][f"{reader}: v7 rubric -> {rubric}"] = moves
    for arm in ARMS:
        first = answers["arms"][arm]["first"]
        out["patterns_per_one_main_job_answer"][arm] = {}
        for group in ("real", POOL):
            counts = [len(patterns_of(first[u["id"]]["reply"])) for u in rows
                      if u["group"] == group and u["labels"][arm] == ONE_MAIN_JOB]
            out["patterns_per_one_main_job_answer"][arm][group] = {
                "answers": len(counts), "mean patterns": float(np.mean(counts)) if counts else None,
                "share listing 2 or more": float(np.mean([c >= 2 for c in counts])) if counts else None}
        real = [u["labels"][arm] for u in rows if u["group"] == "real" and u["labels"][arm] is not None]
        out["essay_shares_vs_real_labels"][arm] = {label: float(np.mean([x == label for x in real])) if real else None
                                                   for label in LABELS[ARMS[arm][1]]}
        pools = [u for u in rows if u["group"] == POOL and u["labels"][arm] == ONE_MAIN_JOB]
        out["examples"][arm] = [{"id": u["id"], "patterns": patterns_of(first[u["id"]]["reply"])}
                                for u in pools[:: max(1, len(pools) // EXAMPLES)][:EXAMPLES]]
    out["essay_shares_vs_real_labels"]["essay (Qwen3-32B neuron poles)"] = ESSAY_SHARES
    return out


def usage(answers: dict) -> dict:
    out = {}
    for arm, parts in answers["arms"].items():
        done = list(parts["first"].values()) + list(parts["retest"].values())
        out[arm] = {"first readings": len(parts["first"]), "retests": len(parts["retest"]),
                    "without a label": sum(a["label"] is None for a in done),
                    "models": sorted({m for a in done for m in a["models"]}),
                    "cost_usd": sum(a["cost_usd"] or 0 for a in done),
                    "input_tokens": sum(a["input_tokens"] for a in done),
                    "output_tokens": sum(a["output_tokens"] for a in done),
                    "mean thinking tokens": float(np.mean([a["thinking_tokens"] for a in done])) if done else None}
    pilot = [a for part in answers["pilot"].values() for a in part.values()]
    out["pilot"] = {"readings": len(pilot), "labeled": sum(a["label"] is not None for a in pilot),
                    "cost_usd": sum(a["cost_usd"] or 0 for a in pilot)}
    out["all CLI calls, by the spend log"] = answers["spend_usd"]
    return out


def analyze(answers: dict, v7_answers: dict, sonnet: dict | None, rng) -> dict:
    base = baselines(v7_answers, sonnet)
    rows = unit_rows(answers, base)
    series = list(ARMS) + list(base)
    summary = {"arms": {s: arm_stats(rows, s, rng) for s in series},
               "versus_v7": {f"{reader}-{rubric}": versus_v7(rows, reader, rubric, rng) if f"{reader}-{V7_RUBRIC}" in base
                             else "not run: no v7-rubric baseline" for reader in READERS for rubric in RUBRICS},
               "e4_versus_e3": {reader: e4_versus_e3(rows, reader, rng) for reader in READERS},
               "retest": {}, "essay_one_main_job": ESSAY_ONE_MAIN_JOB,
               "sonnet_baseline": "v10's first readings" if sonnet else "not available (v10's cache incomplete)",
               "sonnet_baseline_models": sonnet["models"] if sonnet else None}
    for s in series:
        parts = answers["arms"][s] if s in ARMS else None
        first = {k: a["label"] for k, a in parts["first"].items()} if parts else base[s]["first"]
        second = {k: a["label"] for k, a in parts["retest"].items()} if parts else base[s]["retest"]
        if second:
            summary["retest"][s] = {"not one main job": retest(first, second, lambda x: x != ONE_MAIN_JOB, rng)}
            if rubric_of(s) != "E3":
                flag = (lambda x: x == V7_UNRELATED) if rubric_of(s) == V7_RUBRIC else (lambda x: x == UNRELATED_JOBS)
                summary["retest"][s]["unrelated jobs"] = retest(first, second, flag, rng)
    return {"summary": rounded(summary), "exploratory": rounded(exploratory(rows, answers, base)),
            "usage": rounded(usage(answers)), "units": rows}


def main() -> None:
    answers = json.loads(ANSWERS_PATH.read_text())
    v7_answers = json.loads(V7_ANSWERS_PATH.read_text())
    result = analyze(answers, v7_answers, v10_sonnet(v7_answers), np.random.default_rng([16, 1]))
    (RESULTS_DIR / "classification_v16.json").write_text(json.dumps(result, indent=1))
    s = result["summary"]
    for name, arm in s["arms"].items():
        real, pool = arm["one_main_job"]["real"], arm["one_main_job"][POOL]
        det = arm["detection"]["not one main job"]
        print(f"{name}: real one main job {real['value']:.3f} {real['interval']}, 50/50 {pool['value']:.3f} "
              f"{pool['interval']}, detection {det['value']:+.3f} {det['interval']}, AUC {arm['label_auc']['value']:.3f}")
    for name, v in s["versus_v7"].items():
        print(name, v if isinstance(v, str) else f"{v['verdict']}: rise real {v['rise_real']['value']:+.3f} "
              f"{v['rise_real']['interval']}, rise pool {v['rise_pool']['value']:+.3f}, diff "
              f"{v['rise_pool_minus_rise_real']['value']:+.3f} {v['rise_pool_minus_rise_real']['interval']}")
    print(json.dumps(result["usage"], indent=1))


if __name__ == "__main__":
    main()
