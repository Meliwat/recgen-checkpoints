"""Figures for v3: shared-job scores against the controls, help on each facet of the
least-related pair, and label counts."""

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.rubric_v3 import LABELS, MIN_HELP, SAME_JOB


def plot_scores(data, path) -> None:
    neurons = data["neurons"]
    reliable = [(r, key, p) for r in neurons for key, p in r["pairs"].items() if p["both_reliable"]]
    rows = {
        "facet pairs v1 and v2 call related (positive control)":
            [p["score"] for _, _, p in reliable if p["v1_adjacent"] and p["v2_adjacent"]],
        "least-related facet pairs": [p["score"] for r, key, p in reliable if key == r["primary_pair"]],
        "other facet pairs v2 calls unrelated":
            [p["score"] for r, key, p in reliable if key != r["primary_pair"] and not p["v2_adjacent"]],
        "facet vs other neurons' facets, neuron patched in (negative control)": data["negative_control_scores"],
    }
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for y, (name, values) in enumerate(rows.items()):
        alpha = 0.1 if "negative" in name else 0.7
        ax.scatter(values, y + rng.uniform(-0.25, 0.25, len(values)), s=8, alpha=alpha)
        if values:
            ax.plot([np.median(values)] * 2, [y - 0.35, y + 0.35], color="black", linewidth=1.5)
    ax.axvline(SAME_JOB, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(range(len(rows)), [f"{name} (n={len(v)})" for name, v in rows.items()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("shared-job score (1: same use of the neuron beyond its generic push, 0: unrelated)")
    ax.set_title("Facet pairs whose facets both have a reliable signature; black = median, dashed = one-job cutoff",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_primary_help(neurons, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in LABELS:
        points = []
        for r in neurons:
            if r["label"] != label:
                continue
            i, j = map(int, r["primary_pair"].split("-"))
            a, b = sorted((r["facets"][i]["median_help"], r["facets"][j]["median_help"]), reverse=True)
            points.append((a, b, r["neuron"]))
        if points:
            ax.scatter([p[0] for p in points], [p[1] for p in points], s=18, label=f"{label} ({len(points)})")
            for a, b, n in points:
                ax.annotate(str(n), (a, b), fontsize=6, xytext=(2, 2), textcoords="offset points")
    for value in (MIN_HELP, -MIN_HELP):
        ax.axvline(value, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(value, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("median help on the facet it helps more (nats)")
    ax.set_ylabel("median help on the other facet (nats)")
    ax.set_title("Least-related facet pair of each neuron; dashed = least help that counts", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_label_counts(neurons, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    width = 0.38
    groups = {"all traced": neurons, "hand-picked": [r for r in neurons if r["hand_picked"]]}
    for offset, (name, group) in zip((-width / 2, width / 2), groups.items()):
        counts = Counter(r["label"] for r in group)
        ax.barh(np.arange(len(LABELS)) + offset, [counts[label] for label in LABELS], height=width,
                label=f"{name} ({len(group)})")
    ax.set_yticks(np.arange(len(LABELS)), LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("neurons, labeled by their least-related facet pair")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    data = json.loads((RESULTS_DIR / "classification_v3.json").read_text())
    plot_scores(data, FIGURES_DIR / "v3_shared_job.png")
    plot_primary_help(data["neurons"], FIGURES_DIR / "v3_primary_help.png")
    plot_label_counts(data["neurons"], FIGURES_DIR / "v3_labels.png")


if __name__ == "__main__":
    main()
