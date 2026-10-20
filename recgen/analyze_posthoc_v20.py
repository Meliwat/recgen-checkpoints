"""Post hoc v20 analyses, added after an independent review of v20; not pre-registered. analyze_v20 writes them to
results/classification_v20.json under POSTHOC_KEY, apart from the pre-registered results, which they don't change.
The readings are recgen/read_posthoc_v20.py's.

A. same_round: Haiku's T1 and T2's top-50 statistic and paired difference, with v12's top-50 readings replaced by
   this round's reading of the same prompts, so that band and top-50 readings come from one round; the drift between
   the two rounds' top-50 readings; and the 30 top-50 prompts that this round read twice.
B. random_floor: Haiku's flag rate on units of random positions, beside the network reader's random-position
   reference.
C. pools: both readers' flag rates on two-neuron pools of rank-band examples, against the same quartile's single
   neurons, to show whether the readers can detect mixing among rank-band examples, by quartile.
Also the bootstrap interval of the v20 rank-band retest's kappa (retest_kappa).
"""

import numpy as np

from recgen.analyze_v6 import PRIMARY, Layer
from recgen.analyze_v17 import agreement, by_model, label_shares, read_examples
from recgen.read_posthoc_v20 import pool_units, random_units
from recgen.read_v12 import fresh_units
from recgen.record_v20 import v20_path
from recgen.rubric_v5 import MODELS
from recgen.rubric_v6 import cohen_kappa, embed
from recgen.rubric_v7 import UNRELATED_JOBS
from recgen.rubric_v12 import BOOTSTRAP
from recgen.rubric_v20 import DEPTH, PRIMARY_BAND, RANKS, rng_for, t1, t2

POSTHOC_KEY = "posthoc_after_review"
LABEL = "post hoc, added after an independent review; not pre-registered"
PURPOSE = 600  # rng_for purposes from here on, apart from the pre-registered tests'


def flag(label: str | None) -> float | None:
    return None if label is None else float(label == UNRELATED_JOBS)


def mean_interval(values, rng) -> dict:
    """The mean of values with a 95% bootstrap interval."""
    values = np.asarray(values, float)
    draws = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(BOOTSTRAP)]
    return {"n": len(values), "mean": float(values.mean()), "interval": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def kappa_interval(first: list[str], second: list[str], rng) -> list[float]:
    """A 95% bootstrap interval of Cohen's kappa between two readings' flags, resampling units (as v7 did)."""
    a, b = (np.array([label == UNRELATED_JOBS for label in labels]) for labels in (first, second))
    kappas = [cohen_kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(BOOTSTRAP))]
    return [float(x) for x in np.nanpercentile(kappas, [2.5, 97.5])]


def rates(rows: list[dict], field: str) -> list[float]:
    """The mean of field by peak quartile, over rows that have it."""
    return [float(np.mean([r[field] for r in rows if r["quartile"] == q and r.get(field) is not None])) for q in range(4)]


def same_round(haiku: list[dict], answers: dict, v20_answers: dict) -> dict:
    """A. haiku: analyze_v20's Haiku rows (v12's top-50 flag, both rank bands' flags). Each neuron's top-50 flag from
    this round's re-reading replaces v12's in T1 and T2; T2's band statistic is the pre-registered one and isn't
    repeated, and the verdicts depend on it alone."""
    ids = {(m, u["neuron_id"]): u["id"] for m in MODELS for u in fresh_units(m)}
    rows = [{**r, "top_same_round_flag": flag(answers["top_same_round"][ids[r["model"], r["neuron"]]]["label"])}
            for r in haiku]
    rows = [r for r in rows if r["top_same_round_flag"] is not None]
    drift = t1(by_model(rows, "top_same_round_flag", "top_flag"), 1, rng_for(MODELS[0], PURPOSE))
    out = {"n": len(rows),
           "by_quartile": {sample: rates(rows, f"{sample}_flag") for sample in ("top", "top_same_round", "band", "band2")},
           "drift_same_round_minus_v12": {
               "mean_over_models": drift["band_minus_top"], "interval": drift["interval"], "by_model": drift["by_model"],
               "by_quartile": [mean_interval([r["top_same_round_flag"] - r["top_flag"] for r in rows if r["quartile"] == q],
                                             rng_for(MODELS[0], PURPOSE + 1 + q)) for q in range(4)]}}
    for k, band in enumerate(RANKS):
        present = [r for r in rows if r[f"{band}_flag"] is not None]
        purpose = PURPOSE + 10 * (k + 1)
        paired = t2(by_model(present, "log_peak", "quartile", f"{band}_flag", "top_same_round_flag"), 1,
                    rng_for(MODELS[0], purpose + 1))
        out[band] = {"n": len(present),
                     "t1": t1(by_model(present, f"{band}_flag", "top_same_round_flag"), 1, rng_for(MODELS[0], purpose)),
                     "t2_top": paired["top"], "t2_band_minus_top": paired["band_minus_top"]}
    twice = [(v20_answers["top_reread"][uid]["label"], answers["top_same_round"][uid]["label"])
             for uid in v20_answers["top_reread"]]
    first, second = zip(*[pair for pair in twice if None not in pair])
    out["top_read_twice_this_round"] = {**agreement(first, second),
                                        "kappa_interval": kappa_interval(first, second, rng_for(MODELS[0], PURPOSE + 30))}
    return out


def random_floor(answers: dict, network: list[dict]) -> dict:
    """B. Haiku's flag rate and label shares on the random-position units, and the network reader's flag rate and
    coherence on its 50 random positions per neuron (v17's reference), over every neuron with a rank-band sample."""
    labels = [answers["random"][u["id"]]["label"] for m in MODELS for u in random_units(m)]
    labels = [label for label in labels if label is not None]
    return {"haiku_flag": mean_interval([flag(label) for label in labels], rng_for(MODELS[0], PURPOSE + 40)),
            "haiku_label_shares": label_shares(labels),
            "network_reference_flag": mean_interval([r["reference_flag"] for r in network], rng_for(MODELS[0], PURPOSE + 41)),
            "network_reference_coherence": mean_interval([r["reference_coherence"] for r in network],
                                                         rng_for(MODELS[0], PURPOSE + 42))}


def network_pool_flags(model: str, units: list[dict]) -> list[float]:
    """The network reader on each pool's rank-band examples, as v6 and v7 read pools: with both neurons' input
    directions projected out, against the null with both projected out."""
    layer = Layer(model, DEPTH)
    recorded = np.load(v20_path(model))
    z, band_rows, keys = recorded["z"], recorded["band_rows"], recorded["keys"]
    mean, _, directions = layer.embeddings[PRIMARY]
    flags = []
    for u in units:
        (a, b), (from_a, from_b) = u["neurons"], u["picks"]
        rows = np.concatenate([band_rows[a][from_a], band_rows[b][from_b]])
        assert np.array_equal(keys[rows], u["keys"])
        pair = directions[[a, b]]
        flags.append(read_examples(embed(z[rows], mean, pair), layer.nulls[PRIMARY].similarities(pair))["flag"])
    return flags


def excess_by_quartile(pools: list[dict], single: list[dict], rng) -> dict:
    """Pools' flag rate minus the same quartile's single neurons' rank-band rate, by peak quartile, with 95% bootstrap
    intervals resampling pools and neurons within the quartile, and the strongest quartile's excess minus the
    weakest's. pools: {"quartile", "flag", "constituents"} (the mean of its two neurons' own flags); single: rows
    with "quartile" and "band_flag"."""
    pool_flags = [np.array([p["flag"] for p in pools if p["quartile"] == q]) for q in range(4)]
    single_flags = [np.array([r["band_flag"] for r in single if r["quartile"] == q]) for q in range(4)]

    def resampled(x):
        return x[rng.integers(0, len(x), len(x))].mean()

    point = np.array([p.mean() - s.mean() for p, s in zip(pool_flags, single_flags)])
    draws = np.array([[resampled(p) - resampled(s) for p, s in zip(pool_flags, single_flags)] for _ in range(BOOTSTRAP)])
    intervals = np.percentile(draws, [2.5, 97.5], axis=0).T
    return {"pools": [len(p) for p in pool_flags], "single_neurons": [len(s) for s in single_flags],
            "pool_rate": [float(p.mean()) for p in pool_flags], "single_rate": [float(s.mean()) for s in single_flags],
            "constituents_rate": rates(pools, "constituents"),
            "excess": [{"difference": float(d), "interval": [float(lo), float(hi)]} for d, (lo, hi) in zip(point, intervals)],
            "strongest_minus_weakest_excess": {"difference": float(point[3] - point[0]),
                                               "interval": [float(x) for x in np.percentile(draws[:, 3] - draws[:, 0], [2.5, 97.5])]}}


def pools(answers: dict, haiku: list[dict], network: list[dict]) -> dict:
    """C. Both readers on the same two-neuron pools (POOL_PART of ranks 201 to 500 from each of two of v12's fresh
    neurons of one quartile), against single neurons' rank-band flags in that quartile: v12's fresh neurons for
    Haiku, every sampled neuron for the network reader. Also the share of pools that repeat a text window (two
    examples from one window), which v7 found makes pools look less mixed to Haiku."""
    haiku_flags = {(r["model"], r["neuron"]): r["band_flag"] for r in haiku}
    network_by_s = {(r["model"], r["s"]): r for r in network}
    haiku_pools, network_pools, shared_window = [], [], []
    for model in MODELS:
        units = pool_units(model)
        for u, network_flag in zip(units, network_pool_flags(model, units)):
            shared_window.append(len(set(u["keys"][:, 0].tolist())) < len(u["keys"]))
            constituents = [network_by_s[model, s] for s in u["neurons"]]
            assert all(r["quartile"] == u["quartile"] for r in constituents)
            network_pools.append({"quartile": u["quartile"], "flag": network_flag,
                                  "constituents": float(np.mean([r["band_flag"] for r in constituents]))})
            label = answers["pool"][u["id"]]["label"]
            own = [haiku_flags.get((model, r["neuron"])) for r in constituents]
            if label is not None:
                haiku_pools.append({"quartile": u["quartile"], "flag": flag(label),
                                    "constituents": None if None in own else float(np.mean(own))})
    return {"network": excess_by_quartile(network_pools, network, rng_for(MODELS[0], PURPOSE + 50)),
            "haiku": excess_by_quartile(haiku_pools, haiku, rng_for(MODELS[0], PURPOSE + 51)),
            "shared_window_share": float(np.mean(shared_window))}


def posthoc(haiku: list[dict], network: list[dict], v20_answers: dict, answers: dict) -> dict:
    """Every post hoc result. haiku and network: analyze_v20's Haiku rows and network rows with a rank-band sample;
    answers: recgen/read_posthoc_v20.py's."""
    retest = [(v20_answers[PRIMARY_BAND][uid]["label"], a["label"]) for uid, a in v20_answers["retest"].items()]
    first, second = zip(*[pair for pair in retest if None not in pair])
    return {"label": LABEL, "cost_usd": answers["total_cost_usd"],
            "readings": sum(len(answers[k]) for k in ("top_same_round", "random", "pool")),
            "without_label": sum(a["label"] is None for k in ("top_same_round", "random", "pool") for a in answers[k].values()),
            "same_round": same_round(haiku, answers, v20_answers),
            "random_floor": random_floor(answers, network),
            "pools": pools(answers, haiku, network),
            "retest_kappa": {"n": len(first), "kappa": cohen_kappa(*(np.array([x == UNRELATED_JOBS for x in labels])
                                                                     for labels in (first, second))),
                             "interval": kappa_interval(first, second, rng_for(MODELS[0], PURPOSE + 60))}}


def report(block: dict) -> None:
    def interval(x):
        return [round(v, 3) for v in x]

    print(f"{block['label']}: {block['readings']} readings, {block['without_label']} without a label, "
          f"${block['cost_usd']:.2f} CLI-reported")
    a = block["same_round"]
    drift = a["drift_same_round_minus_v12"]
    print(f"A (n={a['n']}): top 50 by quartile, v12 {interval(a['by_quartile']['top'])}, this round "
          f"{interval(a['by_quartile']['top_same_round'])}; drift {drift['mean_over_models']:+.3f} {interval(drift['interval'])}, "
          "by quartile " + "; ".join(f"{q['mean']:+.3f} {interval(q['interval'])}" for q in drift["by_quartile"]))
    for band in RANKS:
        r = a[band]
        print(f"  {band} (n={r['n']}): T1 {r['t1']['band_minus_top']:+.3f} {interval(r['t1']['interval'])} ({r['t1']['reading']}); "
              f"T2 top {r['t2_top']['top_minus_rest']:+.3f} {interval(r['t2_top']['interval'])} ({r['t2_top']['models_negative']}/6), "
              f"band minus top {r['t2_band_minus_top']['difference']:+.3f} {interval(r['t2_band_minus_top']['interval'])}")
    twice = a["top_read_twice_this_round"]
    print(f"  top 50 read twice this round (n={twice['n']}): same flag {twice['same_flag']:.3f}, kappa {twice['kappa']:.2f} "
          f"{interval(twice['kappa_interval'])}")
    b = block["random_floor"]
    print(f"B: Haiku flags {b['haiku_flag']['mean']:.3f} {interval(b['haiku_flag']['interval'])} of {b['haiku_flag']['n']} random units "
          f"(labels {b['haiku_label_shares']}); network reference flag {b['network_reference_flag']['mean']:.3f}, "
          f"coherence {b['network_reference_coherence']['mean']:.1f}")
    print(f"C: {block['pools']['shared_window_share']:.3f} of pools repeat a text window")
    for reader in ("network", "haiku"):
        c = block["pools"][reader]
        print(f"C, {reader}: pools {interval(c['pool_rate'])} (n={c['pools']}), single neurons {interval(c['single_rate'])}, "
              f"constituents {interval(c['constituents_rate'])}; excess " + "; ".join(
                  f"{e['difference']:+.3f} {interval(e['interval'])}" for e in c["excess"])
              + f"; strongest minus weakest {c['strongest_minus_weakest_excess']['difference']:+.3f} "
              f"{interval(c['strongest_minus_weakest_excess']['interval'])}")
    k = block["retest_kappa"]
    print(f"rank-band retest (n={k['n']}): kappa {k['kappa']:.2f} {interval(k['interval'])}")
