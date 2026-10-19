"""Apply the pre-registered v3 rules (recgen/rubric_v3.py) and compare with v1 and v2.

Reads cached data only: results/facets_v3.json (recgen.facets_v3), data/step143000_v3_*
(recgen.outputs), and v1's and v2's results.
"""

import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from recgen.analyze import HAND_PICKED_PATH
from recgen.config import RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.facets import display_text, tokenizer
from recgen.facets_v3 import FACETS_V3_PATH
from recgen.outputs import DELTA_P_PATH, EFFECTS_PATH, OUTPUTS_PATH, PROBABILITY_PATH, facet_examples
from recgen.rubric_v3 import (
    ASSIGN_MIN, HARMLESS, HELPS, LABELS, MIN_EXAMPLES, MIN_HELP, MIN_RELIABILITY, ONE_JOB, SAME_JOB,
    SENSITIVITY_ASSIGN_MIN, SENSITIVITY_MIN_HELP, Signatures, least_related_pair, pair_label, shared_job, spearman_brown,
    use,
)

TOP_TOKENS = 8
REPORT_EXAMPLES = 3
ABLATION = {"zero": 1, "mean": 2}  # variant rows in own_actual: 0 is the neuron as it is


@dataclass(frozen=True)
class Variant:
    ablation: str = "zero"
    min_help: float = MIN_HELP
    assign_min: float = ASSIGN_MIN


PRIMARY = Variant()
SENSITIVITY = {"mean ablation": Variant(ablation="mean"),
               **{f"min help {x}": Variant(min_help=x) for x in SENSITIVITY_MIN_HELP},
               f"assign min {SENSITIVITY_ASSIGN_MIN}": Variant(assign_min=SENSITIVITY_ASSIGN_MIN)}


def load(facets_v3):
    data = dict(np.load(OUTPUTS_PATH))
    assert data["examples"].tolist() == [list(e) for e in facet_examples(facets_v3)], "outputs were recorded for other facets"
    probability = np.load(PROBABILITY_PATH)
    data.update(probability_own=probability["own"], probability_null=probability["null"])
    effects = {kind: np.load(path, mmap_mode="r") for kind, path in EFFECTS_PATH.items()}
    effects["delta_p"] = np.load(DELTA_P_PATH, mmap_mode="r")
    return data, effects


def facet_rows(data, facets_v3, k: int, assign_min: float) -> list[list[int]]:
    """Rows (into the recorded examples) of neuron k's facets, keeping v1's examples and added
    ones at assign_min or above."""
    examples = data["examples"]
    rows = []
    for f, facet in enumerate(facets_v3[k]["facets"]):
        mine = np.flatnonzero((examples[:, 0] == k) & (examples[:, 1] == f))
        rows.append([int(i) for i, e in zip(mine, facet) if e[4] or e[3] >= assign_min])
    return rows


def signatures(data, effects, k: int, ablation: str):
    """Signatures over all of neuron k's recorded examples, and each row's local index."""
    rows = np.flatnonzero(data["examples"][:, 0] == k)
    own = np.asarray(effects[ablation][rows], dtype=np.float32)
    generic = data["generic"][k, 0 if ablation == "zero" else 1]
    local = {int(r): i for i, r in enumerate(rows)}
    return Signatures(own @ own.T, own @ generic, float(generic @ generic)), local


def helps(data, rows, ablation: str, later: bool = False) -> np.ndarray:
    """Help of the neuron at each example: on the next token, or summed over the later predictions."""
    actual = data["own_actual"][rows]
    gain = actual[:, 0] - actual[:, ABLATION[ablation]]
    return gain[:, 1:].sum(axis=1) if later else gain[:, 0]


def neuron_result(data, effects, facets_v3, v1: dict, k: int, variant: Variant) -> dict:
    neuron = facets_v3[k]["neuron"]
    rows = facet_rows(data, facets_v3, k, variant.assign_min)
    sig, local = signatures(data, effects, k, variant.ablation)
    facets = []
    for f, r in enumerate(rows):
        h = helps(data, r, variant.ablation)
        members = [local[i] for i in r]
        facets.append({
            "n": len(r),
            "n_v1": sum(1 for i, e in zip(r, facets_v3[k]["facets"][f]) if e[4]),
            "use": use(h, variant.min_help),
            "median_help": round(float(np.median(h)), 4),
            "help_p": round(float(wilcoxon(h).pvalue), 4) if np.any(h) else 1.0,
            "reliability": round(sig.reliability(members, np.random.default_rng([SEED, neuron, f])), 3)
            if len(r) >= 4 else 0.0,
            "generic_share": round(sig.generic_share(members), 3),
        })
    pairs = {}
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = [local[x] for x in rows[i]], [local[x] for x in rows[j]]
            cosine = sig.specific_cosine(a, b)
            fa, fb = facets[i], facets[j]
            score = (shared_job(cosine, fa["reliability"], fb["reliability"])
                     if min(fa["reliability"], fb["reliability"]) > 0 else float("nan"))
            pairs[f"{i}-{j}"] = {
                "label": pair_label(fa["n"], fb["n"], fa["use"], fb["use"], fa["reliability"], fb["reliability"], score),
                "both_reliable": all(x["n"] >= MIN_EXAMPLES and x["reliability"] >= MIN_RELIABILITY for x in (fa, fb)),
                "score": None if np.isnan(score) else round(score, 3),
                "specific_cosine": round(cosine, 3),
                "raw_cosine": round(sig.raw_cosine(a, b), 3),
            }
    primary = "-".join(map(str, least_related_pair(v1["facet_similarities"])))
    return {"neuron": neuron, "facets": facets, "pairs": pairs, "primary_pair": primary,
            "label": pairs[primary]["label"]}


def exploratory_facets(data, effects, facets_v3, k: int, corpus) -> list[dict]:
    """Per facet: help over the later predictions, the most promoted tokens overall and beyond
    the generic push, and a few examples with their next token and help."""
    tok = tokenizer()
    generic = data["generic"][k, 0]
    unit = generic / np.linalg.norm(generic)
    out = []
    for f, r in enumerate(facet_rows(data, facets_v3, k, ASSIGN_MIN)):
        later = helps(data, r, "zero", later=True)
        mean_effect = np.asarray(effects["zero"][r], dtype=np.float32).mean(axis=0)
        specific = mean_effect - (mean_effect @ unit) * unit
        examples = []
        for i in r[:REPORT_EXAMPLES]:
            _, _, w, p = data["examples"][i]
            examples.append({"text": display_text(corpus, w, p), "next": tok.decode(corpus[w, p + 1:p + 2]),
                             "help": round(float(helps(data, [i], "zero")[0]), 3)})
        moved = np.asarray(effects["delta_p"][r], dtype=np.float32).mean(axis=0)
        tv, kl, sharpen = np.median(data["probability_own"][r], axis=0)
        out.append({
            "median_kl": round(float(kl), 4),
            "median_tv": round(float(tv), 4),
            "median_sharpen": round(float(sharpen), 3),
            "probability_raised": [tok.decode([t]) for t in np.argsort(-moved)[:TOP_TOKENS]],
            "probability_lowered": [tok.decode([t]) for t in np.argsort(moved)[:TOP_TOKENS]],
            "later_use": use(later),
            "median_later_help": round(float(np.median(later)), 4),
            "promoted": [tok.decode([t]) for t in np.argsort(-mean_effect)[:TOP_TOKENS]],
            "specific_promoted": [tok.decode([t]) for t in np.argsort(-specific)[:TOP_TOKENS]],
            "specific_demoted": [tok.decode([t]) for t in np.argsort(specific)[:TOP_TOKENS]],
            "examples": examples,
        })
    return out


def negative_control(data, effects, facets_v3, results: list[dict]) -> list[float]:
    """Shared-job scores between each neuron's reliable facets and other neurons' facets where it
    was patched to its facet-typical activation. Halves are a facet's even and odd examples."""
    counts = data["control_count"]
    facet_ids = [(k, f) for k, r in enumerate(facets_v3) for f in range(len(r["facets"]))]
    scores = []
    for k, result in enumerate(results):
        sig, local = signatures(data, effects, k, "zero")
        generic = data["generic"][k, 0]
        norm2 = float(generic @ generic)
        gram, with_generic = data["control_gram"][k].astype(np.float64), data["control_generic"][k].astype(np.float64)
        others = [x for x, (m, _) in enumerate(facet_ids) if m != k and counts[x].sum() >= MIN_EXAMPLES]
        for f, rows in enumerate(facet_rows(data, facets_v3, k, ASSIGN_MIN)):
            fa = result["facets"][f]
            if fa["n"] < MIN_EXAMPLES or fa["reliability"] < MIN_RELIABILITY:
                continue
            members = [local[i] for i in rows]
            a_generic = float(sig.generic[members].mean())
            a_norm2 = sig.specific_dot(members, members)
            for x in others:
                w = counts[x] / counts[x].sum()
                half_spec = lambda h, g: gram[x, h, g] - with_generic[x, h] * with_generic[x, g] / norm2
                rel_x = spearman_brown(half_spec(0, 1) / np.sqrt(half_spec(0, 0) * half_spec(1, 1)))
                if rel_x < MIN_RELIABILITY:
                    continue
                x_generic = float(w @ with_generic[x])
                x_norm2 = float(w @ gram[x] @ w) - x_generic ** 2 / norm2
                dot = float((data["control_dots"][rows][:, x] @ w).mean()) - a_generic * x_generic / norm2
                scores.append(shared_job(dot / np.sqrt(a_norm2 * x_norm2), fa["reliability"], rel_x))
    return scores


def probability_summary(results) -> dict:
    """Exploratory: how much the neuron changes the next-token prediction on its facets, against
    the same neuron patched into random text, and whether it sharpens the existing prediction."""
    facets = [f for r in results for f in r["facets"] if f["n"] >= MIN_EXAMPLES]
    sign = lambda x: int(np.sign(x))
    primary = [[r["facets"][int(f)]["median_sharpen"] for f in r["primary_pair"].split("-")] for r in results]
    return {
        "median_facet_kl": float(np.median([f["median_kl"] for f in facets])),
        "median_null_kl": float(np.median([r["null_median_kl"] for r in results])),
        "facets_with_kl_above_their_null": sum(f["median_kl"] > r["null_median_kl"] for r in results
                                               for f in r["facets"] if f["n"] >= MIN_EXAMPLES),
        "facets": len(facets),
        "median_facet_tv": float(np.median([f["median_tv"] for f in facets])),
        "median_null_tv": float(np.median([r["null_median_tv"] for r in results])),
        "median_facet_sharpen": float(np.median([f["median_sharpen"] for f in facets])),
        "median_null_sharpen": float(np.median([r["null_median_sharpen"] for r in results])),
        "primary_pairs_sharpen_signs": Counter(f"{sign(a)}/{sign(b)}" if sign(a) >= sign(b) else f"{sign(b)}/{sign(a)}"
                                               for a, b in primary),
    }


def label_counts(results, hand_picked: set[int]) -> dict:
    count = lambda rs: {label: n for label in LABELS if (n := sum(r["label"] == label for r in rs))}
    return {"all": count(results), "hand_picked": count([r for r in results if r["neuron"] in hand_picked])}


def main() -> None:
    corpus = load_corpus()
    facets_v3 = json.loads(FACETS_V3_PATH.read_text())
    v1 = {r["neuron"]: r for r in json.loads((RESULTS_DIR / "classification.json").read_text())["neurons"]}
    v2 = {r["neuron"]: r for r in json.loads((RESULTS_DIR / "classification_v2.json").read_text())["neurons"]}
    hand_picked = {n for n, r in v1.items() if r["hand_picked"]}
    data, effects = load(facets_v3)
    run = lambda variant: [neuron_result(data, effects, facets_v3, v1[r["neuron"]], k, variant)
                           for k, r in enumerate(facets_v3)]
    results = run(PRIMARY)
    sensitivity = {name: label_counts(run(variant), hand_picked) for name, variant in SENSITIVITY.items()}
    for k, r in enumerate(results):
        n = r["neuron"]
        null_tv, null_kl, null_sharpen = np.median(data["probability_null"][:, k], axis=0)
        r.update(null_median_kl=round(float(null_kl), 4), null_median_tv=round(float(null_tv), 4),
                 null_median_sharpen=round(float(null_sharpen), 3))
        r.update(hand_picked=n in hand_picked, v1_label=v1[n]["label"], v2_label=v2[n]["v2_label"],
                 typical_activation=round(float(data["typical"][k]), 3), mean_activation=round(float(data["means"][k]), 4),
                 null_median_help=round(float(np.median(data["null_actual"][:, k, 0, 0] - data["null_actual"][:, k, 1, 0])), 4))
        for facet, extra in zip(r["facets"], exploratory_facets(data, effects, facets_v3, k, corpus)):
            facet.update(extra)
        for key, pair in r["pairs"].items():
            pair["v1_adjacent"] = v2[n]["adjacent_pairs"][key]["v1"]
            pair["v2_adjacent"] = v2[n]["adjacent_pairs"][key]["v2"]

    pairs = [p for r in results for p in r["pairs"].values()]
    control = negative_control(data, effects, facets_v3, results)
    positive = [p["score"] for p in pairs if p["both_reliable"] and p["v1_adjacent"] and p["v2_adjacent"]]
    primary_pairs = [r["pairs"][r["primary_pair"]] for r in results]
    summary = {
        "labels": label_counts(results, hand_picked),
        "labels_v1": {"all": Counter(r["v1_label"] for r in results),
                      "hand_picked": Counter(r["v1_label"] for r in results if r["hand_picked"])},
        "labels_v2": {"all": Counter(r["v2_label"] for r in results),
                      "hand_picked": Counter(r["v2_label"] for r in results if r["hand_picked"])},
        "label_by_v2_label": Counter(f"{r['v2_label']} -> {r['label']}" for r in results),
        "pair_labels": {"all pairs": Counter(p["label"] for p in pairs),
                        "pairs v2 calls unrelated": Counter(p["label"] for p in pairs if not p["v2_adjacent"]),
                        "pairs v1 and v2 call related": Counter(p["label"] for p in pairs
                                                                if p["v1_adjacent"] and p["v2_adjacent"])},
        "facet_uses": Counter(f["use"] for r in results for f in r["facets"]),
        "facet_uses_10_plus": Counter(f["use"] for r in results for f in r["facets"] if f["n"] >= MIN_EXAMPLES),
        "sensitivity": sensitivity,
        "median_facet_size": float(np.median([f["n"] for r in results for f in r["facets"]])),
        "median_reliability": float(np.median([f["reliability"] for r in results for f in r["facets"]])),
        "exploratory": {
            "median_generic_share": float(np.median([f["generic_share"] for r in results for f in r["facets"]])),
            "median_raw_cosine_primary_pairs": float(np.median([p["raw_cosine"] for p in primary_pairs])),
            "median_score_primary_pairs_both_reliable": float(np.median(
                [p["score"] for p in primary_pairs if p["both_reliable"]])),
            "positive_control_scores": sorted(positive),
            "negative_control": {"n": len(control), "median_score": float(np.median(control)),
                                 "share_at_or_above_same_job": float(np.mean(np.array(control) >= SAME_JOB))},
            "later_use_of_facets_harmless_on_next_token": Counter(
                f["later_use"] for r in results for f in r["facets"] if f["use"] == HARMLESS and f["n"] >= MIN_EXAMPLES),
            "median_help_by_use": {u: float(np.median([f["median_help"] for r in results for f in r["facets"]
                                                       if f["use"] == u and f["n"] >= MIN_EXAMPLES]))
                                   for u in sorted({f["use"] for r in results for f in r["facets"] if f["n"] >= MIN_EXAMPLES})},
            "median_null_help": float(np.median([r["null_median_help"] for r in results])),
            "probability": probability_summary(results),
        },
    }
    (RESULTS_DIR / "classification_v3.json").write_text(
        json.dumps({"summary": summary, "neurons": results, "negative_control_scores": [round(x, 3) for x in control]},
                   indent=1))
    write_report(results, json.loads(HAND_PICKED_PATH.read_text()))
    print(json.dumps(summary, indent=1))


def write_report(results, hand_picked) -> None:
    lines = ["# v3 results per neuron\n",
             "Help is the change in log-prob (nats) of the actual next token when the neuron is zero-ablated "
             "at the example (positive: the neuron helps). Reliability is how well two halves of a facet agree on "
             "what the network does with the neuron there beyond its generic push. The shared-job score is about 1 "
             f"when two facets share one job and about 0 when unrelated; one job needs {SAME_JOB} or more. "
             "Promoted tokens are those the neuron raises most on the facet; specific ones are beyond its generic push.\n",
             "| Neuron | Hand-picked | v1 | v2 | v3 (least-related pair) | Median help, A / B | Shared-job score |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        a, b = (r["facets"][int(f)] for f in r["primary_pair"].split("-"))
        lines.append(f"| {r['neuron']} | {'yes' if r['hand_picked'] else ''} | {r['v1_label']} | {r['v2_label']} | "
                     f"{r['label']} | {a['median_help']} / {b['median_help']} | {r['pairs'][r['primary_pair']]['score']} |")
    lines.append("")
    for r in results:
        n = r["neuron"]
        names = hand_picked.get(str(n), {}).get("facets") or [f"facet {i}" for i in range(len(r["facets"]))]
        lines.append(f"## Neuron {n}: v1 {r['v1_label']}, v2 {r['v2_label']}, v3 {r['label']} "
                     f"(facets {r['primary_pair']})\n")
        lines.append(f"Facet-typical activation {r['typical_activation']}; patched into random text at that level, "
                     f"its median help is {r['null_median_help']}, KL {r['null_median_kl']}, sharpen cosine "
                     f"{r['null_median_sharpen']}.\n")
        for i, (name, f) in enumerate(zip(names, r["facets"])):
            lines.append(f"**Facet {i}, {name}** ({f['n']} examples, {f['n_v1']} from v1): {f['use']}, median help "
                         f"{f['median_help']} (p = {f['help_p']}), later predictions {f['median_later_help']} "
                         f"({f['later_use']}); reliability {f['reliability']}, generic share {f['generic_share']}\n")
            lines.append(f"- probability: median KL {f['median_kl']}, moved {f['median_tv']}, sharpen cosine "
                         f"{f['median_sharpen']}; raises {', '.join(repr(t) for t in f['probability_raised'])}; "
                         f"lowers {', '.join(repr(t) for t in f['probability_lowered'])}")
            lines.append(f"- promoted: {', '.join(repr(t) for t in f['promoted'])}")
            lines.append(f"- specific promoted: {', '.join(repr(t) for t in f['specific_promoted'])}")
            lines.append(f"- specific demoted: {', '.join(repr(t) for t in f['specific_demoted'])}")
            lines += [f"- `{e['text']}` → {e['next']!r}, help {e['help']}" for e in f["examples"]]
            lines.append("")
        lines += [f"- pair {key}: {p['label']}; score {p['score']}, specific cosine {p['specific_cosine']}, "
                  f"raw cosine {p['raw_cosine']}" for key, p in r["pairs"].items()]
        lines.append("")
    (RESULTS_DIR / "v3_neurons.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
