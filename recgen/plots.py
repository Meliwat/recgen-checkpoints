"""Figures: each traced neuron's facet responses over training, and the label counts."""

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze import HAND_PICKED_PATH
from recgen.config import CHECKPOINT_STEPS, RESULTS_DIR
from recgen.rubric import (
    MIXED_FROM_START, ONE_ORIGIN_UNRESOLVED, PACKING_LIKE, PRESENT, RECURSIVE_GENERALIZATION, UNSTABLE,
)

FIGURES_DIR = RESULTS_DIR / "figures"
LABEL_ORDER = [RECURSIVE_GENERALIZATION, ONE_ORIGIN_UNRESOLVED, PACKING_LIKE, MIXED_FROM_START, UNSTABLE]
COLUMNS = 4


def step_label(step: int) -> str:
    return f"{step // 1000}k" if step >= 1000 else str(step)


def plot_neuron(ax, result: dict, trajectory: dict, facet_names: list[str] | None) -> None:
    x = np.arange(len(CHECKPOINT_STEPS))
    responses = np.array([trajectory[str(s)]["responses"] for s in CHECKPOINT_STEPS]).T
    names = facet_names or [f"“{t}”" for t in result["top_tokens"]]
    for name, r, emerged in zip(names, responses, result["emergence_steps"]):
        line, = ax.plot(x, r, marker="o", markersize=3, label=name)
        if emerged is not None:
            i = CHECKPOINT_STEPS.index(emerged)
            ax.plot(x[i], r[i], marker="o", markersize=9, markerfacecolor="none", color=line.get_color())
    ax.axhline(PRESENT, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [step_label(s) for s in CHECKPOINT_STEPS], rotation=90, fontsize=7)
    ax.set_ylim(min(-0.1, responses.min()), max(1.3, responses.max()))
    ax.set_title(f"neuron {result['neuron']}: {result['label']}", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")


def plot_grid(results, trajectories, hand_picked, path, title) -> None:
    rows = -(-len(results) // COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(4 * COLUMNS, 3.2 * rows), squeeze=False)
    for ax, result in zip(axes.flat, results):
        names = hand_picked.get(str(result["neuron"]), {}).get("facets")
        plot_neuron(ax, result, trajectories[str(result["neuron"])], names)
    for ax in axes.flat[len(results):]:
        ax.axis("off")
    fig.suptitle(f"{title}\nfacet response (relative to the neuron's top-50 level) by training step; "
                 f"circle = emergence, dashed = present threshold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_label_counts(results, path) -> None:
    groups = {"all traced": results, "hand-picked": [r for r in results if r["hand_picked"]]}
    fig, ax = plt.subplots(figsize=(8, 3))
    width = 0.38
    for offset, (name, members) in zip((-width / 2, width / 2), groups.items()):
        counts = Counter(r["label"] for r in members)
        ax.barh(np.arange(len(LABEL_ORDER)) + offset, [counts[label] for label in LABEL_ORDER],
                height=width, label=f"{name} (n={len(members)})")
    ax.set_yticks(np.arange(len(LABEL_ORDER)), LABEL_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("neurons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    results = json.loads((RESULTS_DIR / "classification.json").read_text())["neurons"]
    trajectories = json.loads((RESULTS_DIR / "trajectories.json").read_text())
    hand_picked = json.loads(HAND_PICKED_PATH.read_text())
    picked = [r for r in results if r["hand_picked"]]
    plot_grid(picked, trajectories, hand_picked, FIGURES_DIR / "hand_picked.png", "Hand-picked mixed-looking neurons")
    plot_grid(results, trajectories, hand_picked, FIGURES_DIR / "all_traced.png", "All traced neurons")
    plot_label_counts(results, FIGURES_DIR / "labels.png")


if __name__ == "__main__":
    main()
