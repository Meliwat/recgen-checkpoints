"""v8 figure: how far above real neurons each reader flags pooled pairs, and how often the
language-model readers call real neurons and pools one job."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v8 import ESSAY_ONE_JOB, LLM_READERS, READERS
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR

NAMES = {"minilm": "MiniLM (v5)", "network": "network (v6)", "haiku": "Haiku, one shot (v7)",
         "haiku agent": "Haiku agent (v8)", "sonnet agent": "Sonnet agent (v8)"}
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"


def dots(ax, y, values, color, label):
    point = np.array([v[0] for v in values])
    low, high = np.array([v[1] for v in values]).T
    ax.errorbar(point, y, xerr=[point - low, high - point], fmt="o", color=color, markersize=6, capsize=3,
                linewidth=1.5, label=label)


def draw(summary: dict, readers: tuple[str, ...], llm_readers: tuple[str, ...], names: dict[str, str], path,
         height: float = 4.2) -> None:
    """Each reader's pooling detection (left) and each language-model reader's one-job shares (right)."""
    excess = summary["pooled_excess"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, height), gridspec_kw={"width_ratios": [1.2, 1]})

    y = np.arange(len(readers))[::-1]
    dots(left, y, [(excess["flag"][r]["mean_over_models"], excess["flag"][r]["interval"]) for r in readers], BLUE,
         "flag = unrelated jobs (C)")
    llm_y = y[[readers.index(r) for r in llm_readers]] - 0.22
    dots(left, llm_y, [(excess["not_one_job"][r]["mean_over_models"], excess["not_one_job"][r]["interval"])
                       for r in llm_readers], ORANGE, "flag = not one job (B or C)")
    left.axvline(0, color=GRAY, linewidth=1)
    left.set_yticks(y, [names[r] for r in readers])
    left.set_xlabel("50/50 pools flagged minus real neurons flagged (mean over 70M, 410M, 2.8B; 95% interval)")
    left.set_title("Detecting two pooled neurons", fontsize=10)
    highest = max(excess[m][r]["interval"][1] for m in excess for r in excess[m])
    left.set_xlim(right=max(0.55, highest + 0.05))
    left.legend(fontsize=8, loc="upper right")

    one_job = summary["one_job_real"]
    y2 = np.arange(len(llm_readers))[::-1]
    dots(right, y2, [(one_job[r]["share"], one_job[r]["interval"]) for r in llm_readers], BLUE, "real neurons")
    pools = summary["one_job_pools"]
    dots(right, y2 - 0.22, [(pools[r]["share"], pools[r]["interval"]) for r in llm_readers], ORANGE, "50/50 pools")
    right.axvline(ESSAY_ONE_JOB, color=GRAY, linestyle="--", linewidth=1)
    right.text(ESSAY_ONE_JOB - 0.01, (y2[0] + y2[1]) / 2, "essay: ~90%\none main job", ha="right", va="center",
               fontsize=8, color="#52514e")
    right.set_yticks(y2, [names[r] for r in llm_readers])
    right.set_xlim(0, 1)
    right.set_xlabel("share labeled one job (A), all three sizes")
    right.set_title("How often a reader says one job", fontsize=10)
    right.legend(fontsize=8, loc="lower right")
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    summary = json.loads((RESULTS_DIR / "classification_v8.json").read_text())["summary"]
    draw(summary, READERS, LLM_READERS, NAMES, FIGURES_DIR / "v8_readers.png")


if __name__ == "__main__":
    main()
