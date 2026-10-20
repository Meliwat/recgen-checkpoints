"""v18 figure: how well each reader separates 50/50 pools from real neurons (label and score AUCs),
the v18 descriptions' simulation scores against v15's explanations on the same units, and the two
Opus agents' label shares."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v18 import V8_READERS
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v8 import NAMES as V8_NAMES
from recgen.plots_v15 import COLORS, GRAY, GRID, INK, LABELS
from recgen.rubric_v8 import MODELS

LABEL_COLORS = {"A": "#86b6ef", "B": "#2a78d6", "C": "#104281"}  # ordinal blue ramp, validated
LABEL_INK = {"A": INK, "B": "white", "C": "white"}
SURFACE = "white"


def auc_panel(ax, summary: dict) -> None:
    v8 = summary["v8 label aucs, for context (unpaired)"]
    primary, scores = summary["primary"], summary["simulation score"]
    rows = [(f"{'flag' if r in ('minilm', 'network') else 'labels'}: {V8_NAMES[r]}", v8[r]["mean_over_models"], v8[r]["interval"], False)
            for r in V8_READERS]
    rows += [("labels: Opus agent (v14)", *primary["v14 label auc"], False),
             ("labels: Opus audit agent (v18)", *primary["v18 label auc"], True),
             ("score: v15 explanation", scores["v15"]["auc, real over pooled 50/50"], scores["v15"]["auc_interval"], False),
             ("score: v18 description", scores["v18"]["auc, real over pooled 50/50"], scores["v18"]["auc_interval"], True)]
    y = np.arange(len(rows))[::-1]
    for yy, (_, point, (low, high), ours) in zip(y, rows):
        ax.errorbar(point, yy, xerr=[[point - low], [high - point]], fmt="o", color=COLORS["real"] if ours else GRAY,
                    markersize=7 if ours else 6, capsize=3, linewidth=1.5)
    ax.axhline(y[-2] + 0.5, color=GRID, linewidth=1)  # labels above, simulation scores below
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1)
    ax.set_ylim(y[-1] - 0.8, y[0] + 0.5)
    ax.text(0.51, y[-1] - 0.55, "chance", fontsize=8, color=GRAY, va="center")
    ax.set_yticks(y, [name for name, *_ in rows], fontsize=8)
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("AUC separating 50/50 pools from real neurons\n(mean over 70M, 410M, 2.8B; 95% interval)")
    ax.set_title("Labels and scores, same units", fontsize=10)


def scores_panel(ax, units: dict) -> None:
    for g in ("real", "pooled 50/50", "related"):
        pairs = np.array([(u["first"]["scores"]["v15"]["pearson"], u["first"]["scores"]["v18"]["pearson"])
                          for m in MODELS for u in units[m] if u["group"] == g
                          and u["first"]["scores"]["v15"] is not None and u["first"]["scores"]["v18"] is not None])
        ax.scatter(*pairs.T, s=14, color=COLORS[g], alpha=0.65, linewidths=0.4, edgecolors=SURFACE, label=LABELS[g])
    ax.plot([-0.4, 1], [-0.4, 1], color=GRAY, linewidth=1, linestyle="--")
    ax.set_xlim(-0.4, 1)
    ax.set_ylim(-0.4, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("v15 explanation's simulation score")
    ax.set_ylabel("v18 description's simulation score")
    ax.legend(fontsize=8, loc="upper left", markerscale=1.4)
    ax.set_title("Same held-out examples, Sonnet simulator", fontsize=10)


def labels_panel(ax, summary: dict) -> None:
    shares = summary["label shares"]
    rows = [(f"{LABELS[g]}, {name}", shares[r][f"{g}, all sizes"]) for g in ("real", "pooled 50/50")
            for r, name in (("v14", "Opus agent (v14)"), ("v18", "audit agent (v18)"))]
    y = np.arange(len(rows))[::-1]
    for yy, (_, share) in zip(y, rows):
        left = 0.0
        for label in ("A", "B", "C"):
            width = share[label]
            ax.barh(yy, width, left=left, height=0.62, color=LABEL_COLORS[label], edgecolor=SURFACE, linewidth=2,
                    label={"A": "A, one job", "B": "B, related jobs", "C": "C, unrelated jobs"}[label] if yy == y[0] else None)
            if width >= 0.08:
                ax.text(left + width / 2, yy, f"{width:.0%}", ha="center", va="center", fontsize=8, color=LABEL_INK[label])
            left += width
    ax.set_yticks(y, [name for name, _ in rows], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of units")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    ax.set_title("Labels by the two Opus agents", fontsize=10)


def main() -> None:
    data = json.loads((RESULTS_DIR / "classification_v18.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained", gridspec_kw={"width_ratios": [1.1, 0.9, 1]})
    auc_panel(axes[0], data["summary"])
    scores_panel(axes[1], data["units"])
    labels_panel(axes[2], data["summary"])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[1].grid(color=GRID, linewidth=0.8)
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v18_auditor.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
