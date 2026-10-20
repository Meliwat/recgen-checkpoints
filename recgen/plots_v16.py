"""v16 figure: how often each reader calls real neurons and known 50/50 pools one (main) job, under
v7's rubric and the essay's two rubrics, beside the essay's ~90%."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v16 import POOL
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.rubric_v16 import ESSAY_ONE_MAIN_JOB, READERS

RUBRIC_NAMES = {"v7": "v7 rubric: one job,\nrelated, unrelated", "E3": "E3, the essay's scheme:\none main job, two related,\nnot yet understood",
                "E4": "E4: E3 plus\nunrelated jobs"}
READER_NAMES = {"haiku": "Haiku", "sonnet": "Sonnet"}
BASELINE_SOURCE = {"haiku": "v7", "sonnet": "v10"}
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"


def dots(ax, y, values, color, label):
    point = np.array([v["value"] for v in values])
    low, high = np.array([v["interval"] for v in values]).T
    ax.errorbar(point, y, xerr=[point - low, high - point], fmt="o", color=color, markersize=6, capsize=3,
                linewidth=1.5, label=label)


def main() -> None:
    arms = json.loads((RESULTS_DIR / "classification_v16.json").read_text())["summary"]["arms"]
    fig, axes = plt.subplots(1, len(READERS), figsize=(12, 4.2), sharey=True)
    rubrics = list(RUBRIC_NAMES)
    for ax, reader in zip(axes, READERS):
        shown = [r for r in rubrics if f"{reader}-{r}" in arms]
        y = np.array([len(rubrics) - 1 - rubrics.index(r) for r in shown], float)
        dots(ax, y + 0.12, [arms[f"{reader}-{r}"]["one_main_job"]["real"] for r in shown], BLUE, "real neurons")
        dots(ax, y - 0.12, [arms[f"{reader}-{r}"]["one_main_job"][POOL] for r in shown], ORANGE, "50/50 pools of two random neurons")
        for r, yy in zip(shown, y):
            auc = arms[f"{reader}-{r}"]["label_auc"]["value"]
            ax.text(0.01, yy + 0.33, f"label AUC {auc:.2f}", fontsize=7.5, color="#52514e", va="center")
        ax.axvline(ESSAY_ONE_MAIN_JOB, color=GRAY, linestyle="--", linewidth=1)
        ax.text(ESSAY_ONE_MAIN_JOB - 0.01, len(rubrics) - 0.55, "essay: ~90%\none main job", ha="right", va="center",
                fontsize=8, color="#52514e")
        ax.set_yticks(np.arange(len(rubrics))[::-1], [RUBRIC_NAMES[r] for r in rubrics], fontsize=8)
        ax.set_ylim(-0.6, len(rubrics) - 0.2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("share called one job (v7) or one main job (E3, E4); mean over six sizes, 95% interval", fontsize=8)
        ax.set_title(f"{READER_NAMES[reader]}, one shot (v7 rubric: {BASELINE_SOURCE[reader]}'s readings)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v16_rubrics.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
