"""Apply the pre-registered v9 rules (recgen/rubric_v9.py): three readers on natural unions of two
neurons, against real neurons and v7's fixed-count pools.

Reads cached data only: v5's window maxima and MiniLM embeddings, v6's recordings, v7's results
and Haiku answers, and v9's Haiku answers (results/v9_llm_answers.json, from recgen.read_v9).
"""

import json
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from recgen.analyze_v6 import PRIMARY as NETWORK_EMBEDDING, Layer, MiniLM, detect
from recgen.config import RESULTS_DIR
from recgen.corpus import load_corpus
from recgen.facets import display_text
from recgen.llm_reader import ANSWERS_PATH as V7_ANSWERS_PATH
from recgen.read_v9 import ANSWERS_PATH
from recgen.rubric_v5 import MODELS
from recgen.rubric_v6 import MERGE_PERCENTILE, cohen_kappa
from recgen.rubric_v7 import DEPTH, UNRELATED_JOBS
from recgen.rubric_v9 import BOOTSTRAP, MINORITY_BINS, SENSITIVITY, mean_excess, minority_bin, rng_for
from recgen.units_v7 import build_units
from recgen.units_v9 import build_unions
from recgen.sweep import sweep_path
from recgen.sweep_v6 import v6_path

READERS = ("minilm", "network", "llm")
EXAMPLES_SHOWN = 2


def network_flag(layer: Layer, rows: np.ndarray, neurons: list[int]) -> bool:
    units, _, levels = layer.embedded(NETWORK_EMBEDDING, rows, neurons)
    return bool(detect(units, levels["merge"][MERGE_PERCENTILE], levels["unrelated"])["unrelated_mixed"])


def flags_for(model: str, minilm: MiniLM, answers: dict, v7_answers: dict) -> dict:
    """Every reader's flag for this model's unions and real neurons, and the detectors' flags for
    the sensitivity unions."""
    layer = Layer(model, DEPTH)
    unions, replaced = build_unions(model)
    out = {"replaced": replaced, "unions": [], "real": [], "sensitivity": {}}
    for u in unions:
        label = answers["first"][u["id"]]["label"]
        out["unions"].append({"id": u["id"], "neuron_ids": u["neuron_ids"], "minority": u["minority"],
                              "sources": u["sources"], "keys": u["keys"],
                              "minilm": bool(minilm.detect(u["keys"])["unrelated_mixed"]),
                              "network": network_flag(layer, u["rows"], list(u["neurons"])),
                              "llm": None if label is None else label == UNRELATED_JOBS, "llm_label": label})
    for u in (x for x in build_units(model) if x["kind"] == "real"):
        label = v7_answers["first"][u["id"]]["label"]
        out["real"].append({"id": u["id"], "minilm": bool(minilm.detect(u["keys"])["unrelated_mixed"]),
                            "network": network_flag(layer, u["rows"], list(u["neurons"])),
                            "llm": None if label is None else label == UNRELATED_JOBS})
    pairs = [tuple(int(x) for x in u["id"].split(":")[-1].split("-")) for u in unions]
    for how in SENSITIVITY:
        variant, dropped = build_unions(model, how, pairs=pairs)
        out["sensitivity"][how] = {"dropped": dropped, "units": [
            {"minority": u["minority"], "minilm": bool(minilm.detect(u["keys"])["unrelated_mixed"]),
             "network": network_flag(layer, u["rows"], list(u["neurons"]))} for u in variant]}
    return out


def flags(rows: list[dict], reader: str) -> np.ndarray:
    return np.array([r[reader] for r in rows if r[reader] is not None], float)


def v7_pools(model: str, v7_units: dict, reader: str) -> np.ndarray:
    return np.array([u[reader] for u in v7_units[model] if u["group"] == "pooled 50/50" and u[reader] is not None], float)


def bin_rates(unions: list[dict], reader: str, rng) -> dict:
    """Flag rate in each minority bin, pooled over models, and the top bin minus the bottom bin."""
    groups = [flags([u for u in unions if minority_bin(u["minority"]) == b], reader) for b in range(len(MINORITY_BINS))]
    draws = [rng.choice(groups[-1], len(groups[-1])).mean() - rng.choice(groups[0], len(groups[0])).mean()
             for _ in range(BOOTSTRAP)]
    return {"bins": [{"minority": f"{low} to {high - 1}", "n": len(g), "share": float(g.mean())}
                     for (low, high), g in zip(MINORITY_BINS, groups)],
            "top_minus_bottom": float(groups[-1].mean() - groups[0].mean()),
            "interval": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def retest(answers: dict) -> dict:
    pairs = [(answers["first"][k]["label"], v["label"]) for k, v in answers["retest"].items()]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    first, second = (np.array([p[i] == UNRELATED_JOBS for p in pairs]) for i in (0, 1))
    return {"units": len(pairs), "same_label": float(np.mean([a == b for a, b in pairs])),
            "same_flag": float(np.mean(first == second)), "flag_kappa": cohen_kappa(first, second)}


def analyze(per_model: dict, v7_units: dict, answers: dict) -> dict:
    unions = [u for m in MODELS for u in per_model[m]["unions"]]
    summary = {"tests": {}, "retest": retest(answers), "replaced": {m: per_model[m]["replaced"] for m in MODELS},
               "minority_counts": dict(sorted(Counter(minority_bin(u["minority"]) for u in unions).items()))}
    for i, reader in enumerate(READERS):
        rng = np.random.default_rng([7, 9, i])
        detection = mean_excess([(flags(per_model[m]["unions"], reader), flags(per_model[m]["real"], reader))
                                 for m in MODELS], rng)
        versus_v7 = mean_excess([(v7_pools(m, v7_units, reader), flags(per_model[m]["unions"], reader))
                                 for m in MODELS], rng)
        summary["tests"][reader] = {
            "union_minus_real": {"mean_over_models": detection[0], "interval": list(detection[1]),
                                 "detects_unions": detection[1][0] > 0},
            "v7_pools_minus_unions": {"mean_over_models": versus_v7[0], "interval": list(versus_v7[1])},
            "by_minority": bin_rates(unions, reader, rng),
            "real_rate": float(np.mean(np.concatenate([flags(per_model[m]["real"], reader) for m in MODELS]))),
            "per_model": {m: {"unions": float(flags(per_model[m]["unions"], reader).mean()),
                              "real": float(flags(per_model[m]["real"], reader).mean())} for m in MODELS},
        }
    summary["sensitivity"] = {}
    for how in SENSITIVITY:
        for i, reader in enumerate(("minilm", "network")):
            point, interval = mean_excess([(flags(per_model[m]["sensitivity"][how]["units"], reader),
                                            flags(per_model[m]["real"], reader)) for m in MODELS],
                                          np.random.default_rng([7, 9, 10 + i]))
            summary["sensitivity"][f"{how}: {reader}"] = {"union_minus_real": point, "interval": list(interval),
                                                          "dropped": sum(per_model[m]["sensitivity"][how]["dropped"] for m in MODELS)}
    return summary


def magnitude_effect(v7_answers: dict) -> dict:
    """Exploratory, found after seeing that unions dominated by one neuron are flagged less than random
    real neurons: flag rate by quartile of a neuron's peak activation (its largest window maximum).
    Network and MiniLM flags are v6's for all sampled neurons; Haiku's are v7's 100 real per model."""
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["neurons"]
    v5 = json.loads((RESULTS_DIR / "classification_v5.json").read_text())["neurons"]
    out = {"network": {}, "minilm": {}, "llm": {}}
    pooled = {"llm": ([], [])}
    for m in MODELS:
        window_max = np.load(sweep_path(m, DEPTH))["max"]
        rows, v5_rows = v6[f"{m} {DEPTH}"], v5[f"{m} {DEPTH}"]
        ids = [int(n) for n in rows]
        peak = window_max[:, ids].astype(np.float32).max(axis=0)
        for reader, values in (("network", [rows[str(i)]["unrelated_mixed"] for i in ids]),
                               ("minilm", [v5_rows[str(i)]["n_facets"] >= 2 and v5_rows[str(i)]["least_related_similarity"] < 0.1
                                           for i in ids])):
            out[reader][m] = quartile_rates(peak, np.array(values, float))
        neurons = np.load(v6_path(m, DEPTH))["neurons"]
        real = [u for u in build_units(m) if u["kind"] == "real"]
        llm_peak = np.array([window_max[:, neurons[u["neurons"][0]]].astype(np.float32).max() for u in real])
        llm_flag = np.array([v7_answers["first"][u["id"]]["label"] == UNRELATED_JOBS for u in real], float)
        out["llm"][m] = quartile_rates(llm_peak, llm_flag)
        pooled["llm"][0].extend(llm_peak / np.median(llm_peak))
        pooled["llm"][1].extend(llm_flag)
    out["llm pooled (peak over model median)"] = quartile_rates(np.array(pooled["llm"][0]), np.array(pooled["llm"][1]))
    return out


def quartile_rates(peak: np.ndarray, flag: np.ndarray) -> dict:
    quartile = np.digitize(peak, np.quantile(peak, [0.25, 0.5, 0.75]))
    return {"by_quartile": [round(float(flag[quartile == q].mean()), 3) for q in range(4)],
            "spearman": round(float(spearmanr(peak, flag).statistic), 3), "n": int(len(flag))}


def write_report(per_model: dict, answers: dict, corpus: np.ndarray) -> None:
    lines = ["# v9 results per model\n",
             "Flag rates of natural unions and of v7's real neurons by reader, then, per model, up to "
             f"{EXAMPLES_SHOWN} balanced unions (10 or more minority examples) Haiku called one job and {EXAMPLES_SHOWN} it "
             "called unrelated jobs, with its stated patterns and two examples from each source neuron.\n"]
    for m in MODELS:
        d = per_model[m]
        lines.append(f"## pythia-{m}\n")
        lines.append("| Reader | unions | real |\n|---|---|---|")
        lines += [f"| {r} | {flags(d['unions'], r).mean():.0%} | {flags(d['real'], r).mean():.0%} |" for r in READERS]
        lines.append("")
        balanced = [u for u in d["unions"] if u["minority"] >= 10]
        for title, chosen in (("Balanced unions Haiku called one job", [u for u in balanced if u["llm_label"] == "A"]),
                              ("Balanced unions Haiku called unrelated jobs", [u for u in balanced if u["llm"]])):
            lines.append(f"### {title}\n")
            for u in chosen[:EXAMPLES_SHOWN]:
                lines.append(f"**{u['id']}**, MLP neurons {u['neuron_ids'][0]} and {u['neuron_ids'][1]}, "
                             f"{u['minority']} of 50 from the minority neuron (MiniLM {u['minilm']}, network {u['network']})\n")
                lines.append(f"- Haiku: `{answers['first'][u['id']]['reply'].replace(chr(10), ' ')[:600]}`")
                for source in (0, 1):
                    picks = [k for k, s in zip(u["keys"], u["sources"]) if s == source][:2]
                    lines += [f"- neuron {u['neuron_ids'][source]}: `{display_text(corpus, int(w), int(p))}`" for w, p in picks]
                lines.append("")
    (RESULTS_DIR / "v9_models.md").write_text("\n".join(lines))


def main() -> None:
    corpus = load_corpus()
    answers = json.loads(ANSWERS_PATH.read_text())
    v7_answers = json.loads(V7_ANSWERS_PATH.read_text())
    v7_units = json.loads((RESULTS_DIR / "classification_v7.json").read_text())["units"]
    minilm = MiniLM(corpus)
    per_model = {m: flags_for(m, minilm, answers, v7_answers) for m in MODELS}
    summary = analyze(per_model, v7_units, answers)
    summary["exploratory_magnitude"] = magnitude_effect(v7_answers)
    units_out = {m: [{k: v for k, v in u.items() if k != "keys"} for u in per_model[m]["unions"]] for m in MODELS}
    (RESULTS_DIR / "classification_v9.json").write_text(json.dumps({"summary": summary, "unions": units_out}, indent=1))
    write_report(per_model, answers, corpus)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
