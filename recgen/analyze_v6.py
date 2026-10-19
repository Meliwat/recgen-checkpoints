"""Apply the pre-registered v6 rules (recgen/rubric_v6.py): a facet detector that reads each
layer's own residual stream, on v5's sampled neurons at each Pythia size and depth, compared
neuron by neuron with v5's MiniLM detector.

Reads cached data only: data/v6/*.npz (recgen.sweep_v6), results/classification_v5.json and
v5's MiniLM embeddings (data/v5/embeddings.npz). MiniLM embeddings of the random positions are
computed once and cached in data/v6/minilm_random.npy.
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from recgen.analyze_v5 import EMBEDDINGS_PATH, rng_for
from recgen.analyze_v5 import least_related as minilm_least_related
from recgen.config import CONTEXT_LEN, RESULTS_DIR, SEED
from recgen.corpus import load_corpus
from recgen.facets import MIN_FACET_SIZE, context_text, display_text
from recgen.facets import embed as minilm_embed
from recgen.facets import find_facets as minilm_find_facets
from recgen.rubric_v5 import AGAINST, DEPTHS, FITS, MODELS, PRIMARY_DEPTH, UNRELATED_SIMILARITY, bootstrap_interval, decide
from recgen.rubric_v6 import (
    EMBEDDING_VARIANTS, MERGE_PERCENTILE, MINILM_PERCENTILES, MIXTURE_PART, N_MIXTURES, SENSITIVITY_SETTINGS,
    UNRELATED_PERCENTILE, cohen_kappa, embed, find_facets, least_related, pair_similarities, partition,
    percentile_rank, random_positions, token_purity, unrelated_mixed,
)
from recgen.sweep_v6 import V6_DIR, sampled_top_examples, v6_path

MINILM_RANDOM_PATH = V6_DIR / "minilm_random.npy"
PRIMARY = "primary"
MEASURES = ("unrelated_mixed", "multi_facet")
MERGE_PERCENTILES = sorted({MERGE_PERCENTILE} | {p for p, _ in SENSITIVITY_SETTINGS.values()}, reverse=True)
EXAMPLE_NEURONS = 3  # v6 unrelated-mixed neurons shown per model and depth in the report
THREADS = 8
POWER_READING = {FITS: "falls with size", AGAINST: "rises with size"}


class NullSpace:
    """Pairwise similarities of the random positions under one embedding (vectors already centered
    and scaled as that embedding does), with a neuron's directions projected out. The same numbers
    as pair_similarities(embed(...)), from one Gram matrix per layer."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float64)
        gram = self.vectors @ self.vectors.T
        self.i, self.j = np.triu_indices(len(vectors), 1)
        self.gram_pairs = gram[self.i, self.j]
        self.norms_sq = np.diag(gram).copy()

    def similarities(self, directions: np.ndarray | None) -> np.ndarray:
        if directions is None:
            inverse = 1 / np.sqrt(self.norms_sq)
            return self.gram_pairs * inverse[self.i] * inverse[self.j]
        basis, _ = np.linalg.qr(np.atleast_2d(directions).astype(np.float64).T)
        projected = self.vectors @ basis
        inverse = 1 / np.sqrt(self.norms_sq - (projected ** 2).sum(axis=1))
        scaled = projected * inverse[:, None]
        return self.gram_pairs * inverse[self.i] * inverse[self.j] - (scaled[self.i] * scaled[self.j]).sum(axis=1)


def levels(null: np.ndarray) -> dict:
    """The merge level at each merge percentile in use, and the unrelated level, from a null."""
    values = np.percentile(null, MERGE_PERCENTILES + [UNRELATED_PERCENTILE])
    return {"merge": dict(zip(MERGE_PERCENTILES, values[:-1].tolist())), "unrelated": float(values[-1])}


def mean_share(vectors: np.ndarray) -> float:
    """Exploratory, added after seeing the uncentered variant: the share of the vectors' mean squared
    norm that lies along their common mean."""
    mean = vectors.mean(axis=0)
    return float(mean @ mean / np.mean(np.sum(vectors ** 2, axis=1)))


def pair_level_unrelated_mixed(units: np.ndarray, facets: list[np.ndarray], unrelated: float) -> bool:
    """Exploratory, added after review: unrelated mixed judged by two facets' mean cross-pair similarity,
    the scale the null is on, instead of their centroids' similarity."""
    return len(facets) >= 2 and min(float((units[a] @ units[b].T).mean())
                                    for a, b in combinations(facets, 2)) < unrelated


def grouped_together(facets: list[np.ndarray], n_examples: int) -> np.ndarray:
    """For each pair of examples (upper triangle), whether they share a facet."""
    labels = partition(facets, n_examples)
    i, j = np.triu_indices(n_examples, 1)
    return labels[i] == labels[j]


def detect(units: np.ndarray, merge: float, unrelated: float, min_size: int = MIN_FACET_SIZE) -> dict:
    facets = find_facets(units, merge, min_size)
    least, pair = least_related(units, facets)
    return {"n_facets": len(facets), "least_related_similarity": least, "least_related_pair": pair,
            "merge_level": merge, "unrelated_level": unrelated,
            "unrelated_mixed": unrelated_mixed(len(facets), least, unrelated), "facets": facets}


class Layer:
    """One model and depth's recorded residuals, and the null of each embedding."""

    def __init__(self, model: str, depth: str):
        data = np.load(v6_path(model, depth))
        self.layer, self.n_layers = int(data["layer"]), int(data["n_layers"])
        self.z, self.keys = data["z"], data["keys"]
        self.neurons, self.rows = data["neurons"], data["example_rows"]
        self.activation_error = float(data["activation_error"])
        weights, gain = data["weights"].astype(np.float64), data["gain"].astype(np.float64)
        random = self.z[data["random_rows"]].astype(np.float64)
        mean = random.mean(axis=0)
        self.mean_share = mean_share(random)
        # embedding -> (mean removed, scale, directions projected out [S, d])
        self.embeddings = {PRIMARY: (mean, None, weights * gain), "with LayerNorm gain": (mean, gain, weights),
                           "uncentered": (None, None, weights * gain), "unprojected": (mean, None, None)}
        assert list(self.embeddings)[1:] == list(EMBEDDING_VARIANTS)
        self.nulls = {name: NullSpace((random - (0 if m is None else m)) * (1 if scale is None else scale))
                      for name, (m, scale, _) in self.embeddings.items()}
        unprojected_null = self.nulls["unprojected"].similarities(None)
        self.unprojected = (unprojected_null, levels(unprojected_null))

    def embedded(self, name: str, rows: np.ndarray, neurons: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
        """Examples at rows under one embedding, with these neurons' directions projected out, and the
        matching null and its levels."""
        mean, scale, directions = self.embeddings[name]
        if directions is None:
            return embed(self.z[rows], mean, None, scale), *self.unprojected
        null = self.nulls[name].similarities(directions[neurons])
        return embed(self.z[rows], mean, directions[neurons], scale), null, levels(null)

    def neuron(self, s: int) -> dict:
        """The v6 detector on sampled neuron s, with v5's other settings and the embedding variants."""
        units, null, lv = self.embedded(PRIMARY, self.rows[s], [s])
        result = detect(units, lv["merge"][MERGE_PERCENTILE], lv["unrelated"])
        result["top_pair_percentile"] = percentile_rank(float(np.median(pair_similarities(units))), null)
        result["pair_level_unrelated_mixed"] = pair_level_unrelated_mixed(units, result["facets"], lv["unrelated"])
        result["sensitivity"] = {setting: detect(units, lv["merge"][percentile], lv["unrelated"], size)["unrelated_mixed"]
                                 for setting, (percentile, size) in SENSITIVITY_SETTINGS.items()}
        result["variants"] = {}
        for name in EMBEDDING_VARIANTS:
            v_units, _, v_levels = self.embedded(name, self.rows[s], [s])
            v = detect(v_units, v_levels["merge"][MERGE_PERCENTILE], v_levels["unrelated"])
            result["variants"][name] = (v["n_facets"], v["unrelated_mixed"])
        return result

    def mixture(self, pair: tuple[int, int]) -> bool:
        """Whether the v6 detector calls two sampled neurons' strongest examples, pooled, unrelated mixed."""
        a, b = pair
        rows = np.concatenate([self.rows[a][:MIXTURE_PART], self.rows[b][:MIXTURE_PART]])
        units, _, lv = self.embedded(PRIMARY, rows, [a, b])
        return detect(units, lv["merge"][MERGE_PERCENTILE], lv["unrelated"])["unrelated_mixed"]


class MiniLM:
    """v5's detector (v1's facets on MiniLM embeddings), read from v5's embedding cache."""

    def __init__(self, corpus: np.ndarray):
        cached = np.load(EMBEDDINGS_PATH)
        self.codes = cached["keys"][:, 0].astype(np.int64) * CONTEXT_LEN + cached["keys"][:, 1]
        assert np.all(np.diff(self.codes) > 0)
        self.vectors = cached["vectors"]
        if not MINILM_RANDOM_PATH.exists():
            V6_DIR.mkdir(exist_ok=True)
            np.save(MINILM_RANDOM_PATH, minilm_embed([context_text(corpus, int(w), int(p))
                                                      for w, p in random_positions(len(corpus))]))
        self.null = pair_similarities(np.load(MINILM_RANDOM_PATH).astype(np.float64))

    def detect(self, keys: np.ndarray) -> dict:
        codes = keys[:, 0].astype(np.int64) * CONTEXT_LEN + keys[:, 1]
        index = np.searchsorted(self.codes, codes)
        assert np.array_equal(self.codes[index], codes), "example missing from v5's embeddings"
        embeddings = self.vectors[index].astype(np.float32)
        facets = minilm_find_facets(embeddings)
        least, pair = minilm_least_related(embeddings, facets)
        return {"n_facets": len(facets), "least_related_similarity": least, "least_related_pair": pair,
                "unrelated_mixed": len(facets) >= 2 and least < UNRELATED_SIMILARITY, "facets": facets,
                "top_pair_percentile": percentile_rank(float(np.median(pair_similarities(embeddings))), self.null)}

    def calibration(self) -> dict:
        """The percentiles of MiniLM's null that rubric_v6 froze, recomputed."""
        return {str(level): round(percentile_rank(level, self.null), 2) for level in MINILM_PERCENTILES}


def mixture_pairs(model: str, depth: str, n_neurons: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng([SEED, MODELS.index(model), list(DEPTHS).index(depth), 1])
    return [tuple(int(x) for x in rng.choice(n_neurons, size=2, replace=False)) for _ in range(N_MIXTURES)]


def median_purity(results: list[dict], keys: np.ndarray, corpus: np.ndarray) -> float:
    """Median token purity over every facet. keys: [S, 50, 2], each neuron's example positions."""
    purities = [token_purity([int(corpus[w, p]) for w, p in keys[s][facet]])
                for s, r in enumerate(results) for facet in r["facets"]]
    return float(np.median(purities)) if purities else float("nan")


def share(flags, rng) -> dict:
    flags = np.asarray(flags, bool)
    low, high = bootstrap_interval(flags, rng)
    return {"share": float(flags.mean()), "interval": [low, high]}


def analyze_layer(model: str, depth: str, minilm: MiniLM, corpus: np.ndarray, published: dict) -> tuple[dict, list, list]:
    """Layer statistics, and the v6 and v5 detectors' results for each sampled neuron."""
    layer = Layer(model, depth)
    keys = layer.keys[layer.rows]
    pairs = mixture_pairs(model, depth, len(layer.neurons))
    with ThreadPoolExecutor(THREADS) as pool:
        v6 = list(pool.map(layer.neuron, range(len(layer.neurons))))
        v6_mixtures = list(pool.map(layer.mixture, pairs))
    v5 = [minilm.detect(k) for k in keys]
    for n, r in zip(layer.neurons, v5):  # rerun from v5's cache, v5's detector gives v5's published facets
        assert r["n_facets"] == published[str(int(n))]["n_facets"]
    v5_mixtures = [minilm.detect(np.concatenate([keys[a][:MIXTURE_PART], keys[b][:MIXTURE_PART]]))["unrelated_mixed"]
                   for a, b in pairs]

    v6_flags, v5_flags = (np.array([r["unrelated_mixed"] for r in rs]) for rs in (v6, v5))
    v6_together, v5_together = (np.concatenate([grouped_together(r["facets"], len(k)) for r, k in zip(rs, keys)])
                                for rs in (v6, v5))
    rng = rng_for(model, depth)  # one generator for the two pre-registered measures, as in v5
    stats = {
        "model": model, "depth": depth, "layer": layer.layer, "n_layers": layer.n_layers, "sampled": len(v6),
        "activation_error": layer.activation_error,
        "unrelated_mixed": share(v6_flags, rng), "multi_facet": share([r["n_facets"] >= 2 for r in v6], rng),
        "facet_counts": dict(sorted(Counter(r["n_facets"] for r in v6).items())),
        "median_merge_level": float(np.median([r["merge_level"] for r in v6])),
        "median_unrelated_level": float(np.median([r["unrelated_level"] for r in v6])),
        "v5_unrelated_mixed_share": float(v5_flags.mean()),
        "v5_multi_facet_share": float(np.mean([r["n_facets"] >= 2 for r in v5])),
        "agreement": {
            "both": int((v6_flags & v5_flags).sum()), "v6_only": int((v6_flags & ~v5_flags).sum()),
            "v5_only": int((~v6_flags & v5_flags).sum()), "neither": int((~v6_flags & ~v5_flags).sum()),
            "share_agreeing": float(np.mean(v6_flags == v5_flags)), "kappa": cohen_kappa(v6_flags, v5_flags),
            "median_ari": float(np.median([adjusted_rand_score(partition(a["facets"], len(k)), partition(b["facets"], len(k)))
                                           for a, b, k in zip(v6, v5, keys)])),
        },
        "checks": {
            "no_facet_share": {"v6": float(np.mean([r["n_facets"] == 0 for r in v6])),
                               "v5": float(np.mean([r["n_facets"] == 0 for r in v5]))},
            "median_top_pair_percentile": {"v6": float(np.median([r["top_pair_percentile"] for r in v6])),
                                           "v5": float(np.median([r["top_pair_percentile"] for r in v5]))},
            "median_token_purity": {"v6": median_purity(v6, keys, corpus), "v5": median_purity(v5, keys, corpus)},
            "mixture_power": {"v6": share(v6_mixtures, rng_for(model, depth)),
                              "v5": share(v5_mixtures, rng_for(model, depth))},
        },
        "sensitivity": {setting_label(s): share([r["sensitivity"][s] for r in v6], rng_for(model, depth))
                        for s in SENSITIVITY_SETTINGS},
        "exploratory_mean_share": layer.mean_share,
        "exploratory_after_review": {
            "v6_flags_among_v5_flagged": float(v6_flags[v5_flags].mean()),
            "v6_flags_among_v5_unflagged": float(v6_flags[~v5_flags].mean()),
            "v6_together_given_v5_together": float(v6_together[v5_together].mean()),
            "v6_together": float(v6_together.mean()),
            "pair_level_unrelated_mixed": share([r["pair_level_unrelated_mixed"] for r in v6], rng_for(model, depth)),
        },
        "variants": {name: {"unrelated_mixed": share([r["variants"][name][1] for r in v6], rng_for(model, depth)),
                            "multi_facet": share([r["variants"][name][0] >= 2 for r in v6], rng_for(model, depth))}
                     for name in EMBEDDING_VARIANTS},
    }
    return stats, [{"neuron": int(n), **r} for n, r in zip(layer.neurons, v6)], v5


def setting_label(setting: tuple[float, int]) -> str:
    distance, min_size = setting
    return f"v5 distance {distance}, minimum facet size {min_size}"


def trend(rows: list[dict]) -> dict:
    """decide() over the six sizes (smallest first) and without 70m. rows: {"share", "interval"} per model."""
    shares, intervals = [r["share"] for r in rows], [tuple(r["interval"]) for r in rows]
    verdict, rho = decide(shares, intervals)
    without, rho_without = decide(shares[1:], intervals[1:])
    return {"shares": [round(s, 3) for s in shares], "verdict": verdict, "spearman_rho": round(rho, 3),
            "verdict_without_70m": without, "spearman_rho_without_70m": round(rho_without, 3)}


def by_model(stats: list[dict], depth: str, get) -> list:
    return [get(next(s for s in stats if s["model"] == m and s["depth"] == depth)) for m in MODELS]


def decisions(stats: list[dict], v5_layers: list[dict], v5_decisions: dict) -> dict:
    """v6's verdicts beside v5's, and whether v5's hold."""
    out = {}
    for depth in DEPTHS:
        for measure in MEASURES:
            v6 = trend(by_model(stats, depth, lambda s: s[measure]))
            v5 = trend(by_model(v5_layers, depth, lambda s: {"share": s[f"{measure}_share"],
                                                             "interval": s[f"{measure}_interval"]}))
            assert v5["verdict"] == v5_decisions[f"{depth}: {measure}"]["verdict"]
            out[f"{depth}: {measure}"] = {
                "primary": depth == PRIMARY_DEPTH and measure == "unrelated_mixed", "v6": v6, "v5": v5,
                "v5_verdict_holds": v6["verdict"] == v5["verdict"],
                "v5_verdict_without_70m_holds": v6["verdict_without_70m"] == v5["verdict_without_70m"]}
    return out


def checks(stats: list[dict]) -> dict:
    """The detector checks by size: power on two-neuron mixtures (with decide()'s reading), and the shares
    of neurons with no facet, top-pair percentiles and token purity."""
    out = {}
    for depth in DEPTHS:
        for detector in ("v6", "v5"):
            power = trend(by_model(stats, depth, lambda s: s["checks"]["mixture_power"][detector]))
            for key in ("verdict", "verdict_without_70m"):
                power[key] = POWER_READING.get(power[key], power[key])
            out[f"{depth}: {detector}"] = {
                "mixture_power": power,
                "mixture_excess_over_single_neurons": [round(x, 3) for x in by_model(stats, depth, lambda s: (
                    s["checks"]["mixture_power"][detector]["share"]
                    - (s["unrelated_mixed"]["share"] if detector == "v6" else s["v5_unrelated_mixed_share"])))],
                **{check: [round(x, 3) for x in by_model(stats, depth, lambda s: s["checks"][check][detector])]
                   for check in ("no_facet_share", "median_top_pair_percentile", "median_token_purity")}}
    return out


def after_review(stats: list[dict]) -> dict:
    """Exploratory checks added after the independent review, by depth: the pair-level cutoff's trend, how
    often v6 groups together the examples MiniLM groups, and v6's flags among neurons v5 does and doesn't flag."""
    out = {}
    for depth in DEPTHS:
        row = {"pair_level_unrelated_mixed": trend(by_model(stats, depth, lambda s: s["exploratory_after_review"]["pair_level_unrelated_mixed"]))}
        for key in ("v6_together_given_v5_together", "v6_together", "v6_flags_among_v5_flagged", "v6_flags_among_v5_unflagged"):
            row[key] = [round(x, 3) for x in by_model(stats, depth, lambda s: s["exploratory_after_review"][key])]
        out[depth] = row
    return out


def sensitivity(stats: list[dict]) -> dict:
    out = {}
    for depth in DEPTHS:
        for setting in SENSITIVITY_SETTINGS:
            label = setting_label(setting)
            out[f"{depth}: {label}"] = trend(by_model(stats, depth, lambda s: s["sensitivity"][label]))
        for name in EMBEDDING_VARIANTS:
            for measure in MEASURES:
                out[f"{depth}: {name}, {measure}"] = trend(by_model(stats, depth, lambda s: s["variants"][name][measure]))
    return out


def write_report(stats: list[dict], examples: dict, corpus: np.ndarray) -> None:
    lines = ["# v6 results per model\n",
             "Share of sampled neurons that the network-native detector calls unrelated mixed and that have 2+ facets, "
             "with 95% bootstrap intervals, beside v5's MiniLM detector on the same neurons and the two detectors' "
             "agreement. Then, at each depth, "
             f"{EXAMPLE_NEURONS} randomly chosen v6 unrelated-mixed neurons with two examples per facet.\n"]
    for model in MODELS:
        lines.append(f"## pythia-{model}\n")
        lines.append("| Depth | Layer | Sampled | v6 unrelated mixed | v6 2+ facets | v5 unrelated mixed | v5 2+ facets "
                     "| Flags agree | Kappa | Median ARI |\n|---|---|---|---|---|---|---|---|---|---|")
        for s in (s for s in stats if s["model"] == model):
            (u, (ulo, uhi)), (m, (mlo, mhi)) = ((s[k]["share"], s[k]["interval"]) for k in MEASURES)
            a = s["agreement"]
            lines.append(f"| {s['depth']} | {s['layer']} of {s['n_layers']} | {s['sampled']} | "
                         f"{u:.3f} ({ulo:.3f} to {uhi:.3f}) | {m:.3f} ({mlo:.3f} to {mhi:.3f}) | "
                         f"{s['v5_unrelated_mixed_share']:.3f} | {s['v5_multi_facet_share']:.3f} | "
                         f"{a['share_agreeing']:.3f} | {a['kappa']:.2f} | {a['median_ari']:.2f} |")
        lines.append("")
        for depth in DEPTHS:
            lines.append(f"### {depth} depth\n")
            for r, v5 in examples[model, depth]:
                lines.append(f"**Neuron {r['neuron']}** ({r['n_facets']} facets, least-related similarity "
                             f"{r['least_related_similarity']:.2f}, unrelated below {r['unrelated_level']:.2f}; "
                             f"v5: {v5['n_facets']} facets, {'unrelated mixed' if v5['unrelated_mixed'] else 'not unrelated mixed'})\n")
                for i, facet in enumerate(r["facets"]):
                    lines += [f"- facet {i}: {a:.2f} `{display_text(corpus, w, p)}`" for w, p, a in facet[:2]]
                lines.append("")
    (RESULTS_DIR / "v6_models.md").write_text("\n".join(lines))


def report_examples(model: str, depth: str, v6: list[dict], v5: list[dict], keys: np.ndarray, acts: np.ndarray) -> list:
    """EXAMPLE_NEURONS random v6 unrelated-mixed neurons, each facet as (window, position, activation), strongest first."""
    mixed = [i for i, r in enumerate(v6) if r["unrelated_mixed"]]
    shown = rng_for(model, depth).choice(mixed, size=min(EXAMPLE_NEURONS, len(mixed)), replace=False)
    out = []
    for i in sorted(int(x) for x in shown):
        facets = [[(int(keys[i][j][0]), int(keys[i][j][1]), float(acts[i][j])) for j in sorted(f)] for f in v6[i]["facets"]]
        out.append(({**v6[i], "facets": facets}, v5[i]))
    return out


def rounded(x):
    return None if x is None else round(float(x), 4)


def main() -> None:
    corpus = load_corpus()
    v5_results = json.loads((RESULTS_DIR / "classification_v5.json").read_text())
    minilm = MiniLM(corpus)
    stats, neurons_out, examples = [], {}, {}
    for model in MODELS:
        for depth in DEPTHS:
            s, v6, v5 = analyze_layer(model, depth, minilm, corpus, v5_results["neurons"][f"{model} {depth}"])
            stats.append(s)
            _, keys, acts, _, _ = sampled_top_examples(model, depth)
            examples[model, depth] = report_examples(model, depth, v6, v5, keys, acts)
            neurons_out[f"{model} {depth}"] = {
                str(r["neuron"]): {"n_facets": r["n_facets"], "least_related_similarity": rounded(r["least_related_similarity"]),
                                   "unrelated_level": rounded(r["unrelated_level"]), "unrelated_mixed": r["unrelated_mixed"],
                                   "top_pair_percentile": rounded(r["top_pair_percentile"]),
                                   "v5_n_facets": m["n_facets"], "v5_unrelated_mixed": m["unrelated_mixed"]}
                for r, m in zip(v6, v5)}
            print(f"pythia-{model} {depth}: v6 unrelated mixed {s['unrelated_mixed']['share']:.3f}, "
                  f"v5 {s['v5_unrelated_mixed_share']:.3f}, kappa {s['agreement']['kappa']:.2f}", flush=True)
    calibration = minilm.calibration()
    summary = {
        "decisions": decisions(stats, v5_results["layers"], v5_results["summary"]["decisions"]),
        "detector_checks": checks(stats),
        "sensitivity": sensitivity(stats),
        "minilm_calibration": {"frozen": {str(k): v for k, v in MINILM_PERCENTILES.items()}, "recomputed": calibration,
                               "matches": all(abs(calibration[str(k)] - v) <= 0.05 for k, v in MINILM_PERCENTILES.items())},
        "largest_activation_error": max(s["activation_error"] for s in stats),
        "exploratory_mean_share": {depth: [round(x, 3) for x in by_model(stats, depth, lambda s: s["exploratory_mean_share"])]
                                   for depth in DEPTHS},
        "exploratory_after_review": after_review(stats),
    }
    (RESULTS_DIR / "classification_v6.json").write_text(
        json.dumps({"summary": summary, "layers": stats, "neurons": neurons_out}, indent=1))
    write_report(stats, examples, corpus)
    print(json.dumps({k: summary[k] for k in ("decisions", "minilm_calibration", "largest_activation_error")}, indent=1))


if __name__ == "__main__":
    main()
