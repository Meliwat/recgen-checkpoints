"""v13 figures: label counts beside v4's, and the unrelated facet pairs that branched then separated."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.config import RESULTS_DIR
from recgen.plots import COLUMNS, FIGURES_DIR, LABEL_ORDER
from recgen.plots_v4 import plot_pair


def plot_branched(results: list[dict], path) -> None:
    """Every unrelated pair that was adjacent at formation (MLP drivers) and not at the end."""
    pairs = [(r, key) for r in results for key, p in r["mlp"]["pairs"].items() if p.get("branched_then_separated")]
    rows = -(-len(pairs) // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(4 * COLUMNS, 3.2 * rows), squeeze=False)
    for ax, (r, key) in zip(axes.flat, pairs):
        plot_pair(ax, r, key, None)
        ax.set_title(f"neuron {r['neuron']}, facets {key.replace('-', ' and ')}: {r['mlp']['label']}", fontsize=9)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    fig.suptitle("v13: unrelated facet pairs adjacent at formation but not at the end (branched then separated)\n"
                 "share of top-100 upstream MLP drivers in common, by training step", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_labels(v4_labels: dict, summary: dict, path) -> None:
    versions = {"v4, MiniLM facets (MLP drivers)": v4_labels, "v13, network facets (MLP drivers)": summary["mlp"]["labels"],
                "v13, network facets (heads)": summary["heads"]["labels"]}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    width = 0.8 / len(versions)
    for v, (name, counts) in enumerate(versions.items()):
        ax.barh(np.arange(len(LABEL_ORDER)) + (v - 1) * width, [counts.get(label, 0) for label in LABEL_ORDER],
                height=width, label=name)
    ax.set_yticks(np.arange(len(LABEL_ORDER)), LABEL_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("neurons (40 per version)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    data = json.loads((RESULTS_DIR / "classification_v13.json").read_text())
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_labels(data["summary"]["v4_for_comparison"]["labels"], data["summary"], FIGURES_DIR / "v13_labels.png")
    plot_branched(data["neurons"], FIGURES_DIR / "v13_branched.png")


if __name__ == "__main__":
    main()
