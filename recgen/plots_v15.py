"""v15 figure: simulation scores by unit type and size for both simulators, and how well the score
separates 50/50 pools from real neurons beside v8's label readers."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v15 import ARMS, GROUPS, V8_READERS
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v8 import NAMES as V8_NAMES
from recgen.rubric_v8 import MODELS

COLORS = {"real": "#2a78d6", "pooled 50/50": "#eb6834", "related": "#1f9e89"}  # validated: CVD and contrast pass
LABELS = {"real": "real neurons", "pooled 50/50": "50/50 pools", "related": "related pools"}
ARM_NAMES = {"sonnet": "Sonnet", "haiku": "Haiku"}
INK, GRAY, GRID = "#30302e", "#8a8984", "#e6e5e0"


def scores_panel(ax, units: dict, arm: str) -> None:
    rng = np.random.default_rng(0)
    for i, model in enumerate(MODELS):
        for j, g in enumerate(GROUPS):
            x = i * 4 + j
            values = [u["first"]["scores"][arm]["pearson"] for u in units[model]
                      if u["group"] == g and u["first"]["scores"][arm] is not None]
            ax.scatter(x + rng.uniform(-0.28, 0.28, len(values)), values, s=12, color=COLORS[g], alpha=0.6, linewidths=0,
                       label=LABELS[g] if i == 0 else None)
            ax.plot([x - 0.36, x + 0.36], [np.mean(values)] * 2, color=INK, linewidth=2)
    ax.set_xticks([i * 4 + 1 for i in range(len(MODELS))], [f"pythia-{m}" for m in MODELS])
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_title(f"Explainer Sonnet, simulator {ARM_NAMES[arm]} (bar: mean)", fontsize=10)


def auc_panel(ax, summary: dict, v8_auc: dict) -> None:
    rows = [(f"score, {ARM_NAMES[arm]} simulator (v15)",
             (summary["primary"] if arm == "sonnet" else summary["other simulator"][arm]), True) for arm in ARMS]
    rows += [(f"labels: {V8_NAMES[r]}", v8_auc[r], False) for r in V8_READERS]
    y = np.arange(len(rows))[::-1]
    for yy, (name, stats, ours) in zip(y, rows):
        point = stats["auc, real over pooled 50/50"] if ours else stats["mean_over_models"]
        low, high = stats["auc_interval"] if ours else stats["interval"]
        ax.errorbar(point, yy, xerr=[[point - low], [high - point]], fmt="o", color=COLORS["real"] if ours else GRAY,
                    markersize=6, capsize=3, linewidth=1.5)
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1)
    ax.set_ylim(y[-1] - 0.8, y[0] + 0.5)
    ax.text(0.51, y[-1] - 0.55, "chance", fontsize=8, color=GRAY, va="center")
    ax.set_yticks(y, [name for name, *_ in rows], fontsize=8)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("AUC separating 50/50 pools from real neurons\n(mean over 70M, 410M, 2.8B; 95% interval)")
    ax.set_title("Score vs v8's label readers, same units", fontsize=10)


def main() -> None:
    data = json.loads((RESULTS_DIR / "classification_v15.json").read_text())
    v8_auc = json.loads((RESULTS_DIR / "classification_v8.json").read_text())["summary"]["post_hoc"]["label_auc"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, arm in zip(axes[:2], ARMS):
        scores_panel(ax, data["units"], arm)
    axes[1].sharey(axes[0])
    axes[0].set_ylabel("simulation score (Pearson r, 60 held-out examples)")
    axes[0].legend(fontsize=8, loc="lower left", markerscale=1.5)
    auc_panel(axes[2], data["summary"], v8_auc)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[2] else "y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v15_simulation.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
