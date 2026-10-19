"""Figures for v2: driver overlaps against their null and ceiling, bridge gains over training,
and v1 against v2 label counts."""

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.config import RESULTS_DIR
from recgen.plots import COLUMNS, FIGURES_DIR, LABEL_ORDER, step_label
from recgen.rubric_v2 import MIN_BRIDGE_GAIN

TITLE_INCHES = 0.7  # room above the subplot grid for the two-line title


def plot_overlaps(neurons, path) -> None:
    rows = {
        "two halves of one facet (ceiling)": [x for r in neurons for x in r["split_half_overlaps"]],
        "facet pairs v1 called adjacent": [r["driver_overlaps"][k] for r in neurons
                                           for k, a in r["adjacent_pairs"].items() if a["v1"]],
        "facet pairs v1 called unrelated": [r["driver_overlaps"][k] for r in neurons
                                            for k, a in r["adjacent_pairs"].items() if not a["v1"]],
        "facet vs other neurons' facets (null)": [x for r in neurons for x in r["null_overlaps"]],
    }
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for y, (name, values) in enumerate(rows.items()):
        alpha = 0.05 if "null" in name else 0.7
        ax.scatter(values, y + rng.uniform(-0.25, 0.25, len(values)), s=8, alpha=alpha)
        ax.plot([np.median(values)] * 2, [y - 0.35, y + 0.35], color="black", linewidth=1.5)
    threshold = np.median([r["adjacency_threshold"] for r in neurons])
    ax.axvline(threshold, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(rows)), [f"{name} (n={len(v)})" for name, v in rows.items()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of top-100 upstream drivers in common")
    ax.set_title("Upstream driver overlap; black = median, dashed = median adjacency threshold", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_bridges(bridges, path, title) -> None:
    rows = -(-len(bridges) // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(4 * COLUMNS, 3 * rows), squeeze=False)
    for ax, b in zip(axes.flat, bridges):
        x = np.arange(len(b["steps"]))
        for name in ("facet", "bridge", "placebo"):
            ax.plot(x, b["gains"][name], marker="o", markersize=3, label=name)
        ax.axhline(MIN_BRIDGE_GAIN * b["gains"]["facet"][-1], color="grey", linestyle="--", linewidth=0.8)
        ax.set_xticks(x, [step_label(s) for s in b["steps"]], rotation=90, fontsize=7)
        ax.set_title(f"neuron {b['neuron']}, facet {b['a']} → {b['b']}: {b['bridge']}", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
    for ax in axes.flat[len(bridges):]:
        ax.axis("off")
    fig.suptitle(f"{title}\npre-activation gain from the earlier facet's emergence to the later one's; "
                 f"dashed = least gain a bridge needs by the end", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - TITLE_INCHES / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_gain_shares(bridges, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    shares = {name: [b["gains"][name][-1] / b["gains"]["facet"][-1] for b in bridges] for name in ("bridge", "placebo")}
    ax.hist(list(shares.values()), bins=np.linspace(-0.3, 0.4, 29), label=[f"{k} (n={len(v)})" for k, v in shares.items()])
    ax.axvline(MIN_BRIDGE_GAIN, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("gain when the later facet emerges, as a share of its own gain")
    ax.set_ylabel("facet pairs")
    ax.set_title("Bridge and placebo gains; dashed = least share a bridge needs", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_label_counts(neurons, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    width = 0.38
    for offset, version in zip((-width / 2, width / 2), ("v1", "v2")):
        counts = Counter(r[f"{version}_label"] for r in neurons)
        ax.barh(np.arange(len(LABEL_ORDER)) + offset, [counts[label] for label in LABEL_ORDER],
                height=width, label=f"{version} adjacency")
    ax.set_yticks(np.arange(len(LABEL_ORDER)), LABEL_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel(f"neurons (of {len(neurons)} traced)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    data = json.loads((RESULTS_DIR / "classification_v2.json").read_text())
    neurons, bridges = data["neurons"], data["bridges"]
    hand_picked = {r["neuron"] for r in neurons if r["hand_picked"]}
    plot_overlaps(neurons, FIGURES_DIR / "v2_driver_overlaps.png")
    plot_bridges([b for b in bridges if b["neuron"] in hand_picked], FIGURES_DIR / "v2_bridges_hand_picked.png",
                 "Bridge search, hand-picked neurons")
    plot_bridges(bridges, FIGURES_DIR / "v2_bridges_all.png", "Bridge search, all facet pairs")
    plot_gain_shares(bridges, FIGURES_DIR / "v2_bridge_gain_shares.png")
    plot_label_counts(neurons, FIGURES_DIR / "v2_labels.png")


if __name__ == "__main__":
    main()
