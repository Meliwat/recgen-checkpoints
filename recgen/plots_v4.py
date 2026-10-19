"""v4 figures: driver overlap over training for each hand-picked neuron's least-related pair,
label counts across versions, and where the facets' rise comes from over training."""

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze import HAND_PICKED_PATH
from recgen.analyze_v4 import least_related_pair
from recgen.config import RESULTS_DIR
from recgen.plots import COLUMNS, FIGURES_DIR, LABEL_ORDER, step_label
from recgen.rubric_v4 import STEPS


def plot_pair(ax, result: dict, key: str, names: list[str] | None) -> None:
    x = np.arange(len(STEPS))
    pair = result["mlp"]["pairs"][key]
    i, j = map(int, key.split("-"))
    label = f"{names[i]} / {names[j]}" if names else f"facets {i} and {j}"
    ax.plot(x, pair["series"], marker="o", markersize=3, label=label)
    if pair.get("window"):
        start, end = STEPS.index(pair["window"][0]), STEPS.index(pair["window"][1])
        ax.axvspan(start - 0.5, end + 0.5, color="grey", alpha=0.15, label="formation window")
        ax.plot(x, pair["null_p95_series"], color="grey", linestyle="--", linewidth=0.8, label="null, 95th percentile")
        ax.hlines(pair["threshold"], start - 0.5, end + 0.5, color="black", linewidth=1.2, label="threshold (window peak)")
    ax.set_xticks(x, [step_label(s) for s in STEPS], rotation=90, fontsize=7)
    ax.set_ylim(0, max(0.3, max(pair["series"]) * 1.1))
    ax.set_title(f"neuron {result['neuron']}: v4 {result['mlp']['label']}", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")


def plot_hand_picked(results, v1_rows, hand_picked, path) -> None:
    picked = [r for r in results if r["hand_picked"]]
    rows = -(-len(picked) // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(4 * COLUMNS, 3.2 * rows), squeeze=False)
    for ax, r in zip(axes.flat, picked):
        plot_pair(ax, r, least_related_pair(v1_rows[r["neuron"]]), hand_picked.get(str(r["neuron"]), {}).get("facets"))
    for ax in axes.flat[len(picked):]:
        ax.axis("off")
    fig.suptitle("Least-related facet pair: share of top-100 upstream MLP drivers in common, by training step\n"
                 "adjacent at formation if the peak inside the window beats the threshold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # leave room for the two-line title
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_labels(results, path) -> None:
    versions = {"v1": lambda r: r["v1_label"], "v2": lambda r: r["v2_label"],
                "v4 (MLP drivers)": lambda r: r["mlp"]["label"], "v4 (heads)": lambda r: r["heads"]["label"]}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    width = 0.8 / len(versions)
    for v, (name, get) in enumerate(versions.items()):
        counts = Counter(get(r) for r in results)
        ax.barh(np.arange(len(LABEL_ORDER)) + (v - 1.5) * width, [counts[label] for label in LABEL_ORDER],
                height=width, label=name)
    ax.set_yticks(np.arange(len(LABEL_ORDER)), LABEL_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("neurons (all 40 traced)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_sources(summary, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(STEPS))
    for part, series in summary["median_source_share_by_step"].items():
        ax.plot(x, [series[str(s)] for s in STEPS], marker="o", markersize=3, label=part.replace("_", " "))
    ax.set_xticks(x, [step_label(s) for s in STEPS], rotation=90, fontsize=7)
    ax.set_ylabel("median share of a facet's rise")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    data = json.loads((RESULTS_DIR / "classification_v4.json").read_text())
    v1_rows = {r["neuron"]: r for r in json.loads((RESULTS_DIR / "classification.json").read_text())["neurons"]}
    hand_picked = json.loads(HAND_PICKED_PATH.read_text())
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_hand_picked(data["neurons"], v1_rows, hand_picked, FIGURES_DIR / "v4_formation_overlap.png")
    plot_labels(data["neurons"], FIGURES_DIR / "v4_labels.png")
    plot_sources(data["summary"], FIGURES_DIR / "v4_sources.png")


if __name__ == "__main__":
    main()
