"""v24 figure: how often each reader and rubric calls random text, real neurons and known 50/50 pools one (main)
job and C (unrelated jobs, or not yet understood), all read in one run, beside the essay's ~90%."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v16 import BLUE, GRAY, ORANGE, dots
from recgen.rubric_v24 import ARMS, C_LABEL, E3, ONE_JOB, POOL, RANDOM, REAL

ESSAY = 0.90
GREEN = "#1f9e74"
NAMES = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus"}
RUBRIC_NAMES = {"v7": "v7 rubric", E3: "E3"}
GROUP_STYLE = ((RANDOM, GREEN, "random text (50 random positions)", 0.2), (REAL, BLUE, "real neurons", 0.0),
               (POOL, ORANGE, "50/50 pools of two random neurons", -0.2))


def panel(ax, series: dict, label: str, title: str, essay: bool) -> None:
    y = np.arange(len(series))[::-1].astype(float)
    for group, color, name, offset in GROUP_STYLE:
        dots(ax, y + offset, [st["shares"][group][label] for st in series.values()], color, name)
    if essay:
        ax.axvline(ESSAY, color=GRAY, linestyle="--", linewidth=1)
        ax.text(ESSAY - 0.01, len(series) - 0.45, "essay: ~90%\none main job", ha="right", va="center", fontsize=8,
                color="#52514e")
    else:
        for st, yy in zip(series.values(), y):
            aucs = st["label_auc"]
            ax.text(1.0, yy + 0.38, f"AUC random>real {aucs['random over real']['value']:.2f}, "
                    f"pools>real {aucs['pools over real']['value']:.2f}", fontsize=7, color="#52514e", ha="right", va="center")
    ax.set_yticks(y, [f"{NAMES[s.split('-')[0]]}, {RUBRIC_NAMES[s.split('-')[1]]}" for s in series], fontsize=8)
    ax.set_ylim(-0.6, len(series) - 0.1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("share of units, mean over sizes, 95% interval", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    summary = json.loads((RESULTS_DIR / "classification_v24.json").read_text())["summary"]
    series = {s: summary["series"][s] for s in ARMS}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panel(axes[0], series, ONE_JOB, "A: one job (v7 rubric) or one main job (E3)", essay=True)
    panel(axes[1], series, C_LABEL, "C: unrelated jobs (v7 rubric) or not yet understood (E3)", essay=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v24_random_text.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
