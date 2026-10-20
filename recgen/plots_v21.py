"""v21 figure: how often Opus calls real neurons and known 50/50 pools one (main) job under v7's rubric
and the essay's E3, one shot (arm A, beside v16's readers) and as an investigating agent (arm B,
beside v14's), with each row's label AUC and the essay's ~90%."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v21 import OPUS_E3
from recgen.analyze_v16 import POOL
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v16 import BLUE, GRAY, ORANGE, dots
from recgen.rubric_v21 import ARM_B, ESSAY_ONE_MAIN_JOB, V14_AGENT

ARM_A_ROWS = {"opus-v7": "Opus, v7 rubric", "opus-E3": "Opus, E3", "sonnet-E3": "Sonnet, E3 (v16)",
              "haiku-E3": "Haiku, E3 (v16)"}
ARM_B_ROWS = {V14_AGENT: "Opus agent,\nv7 rubric (v14)", ARM_B: "Opus agent, E3", "one shot": "Opus one shot, E3\n(same units)"}


def panel(ax, rows: dict[str, dict], names: dict[str, str], title: str) -> None:
    y = np.arange(len(rows))[::-1].astype(float)
    dots(ax, y + 0.12, [r["one_main_job"]["real"] for r in rows.values()], BLUE, "real neurons")
    dots(ax, y - 0.12, [r["one_main_job"][POOL] for r in rows.values()], ORANGE, "50/50 pools of two random neurons")
    for row, yy in zip(rows.values(), y):
        if "label_auc" in row:
            ax.text(0.01, yy + 0.33, f"label AUC {row['label_auc']['value']:.2f}", fontsize=7.5, color="#52514e", va="center")
    ax.axvline(ESSAY_ONE_MAIN_JOB, color=GRAY, linestyle="--", linewidth=1)
    ax.text(ESSAY_ONE_MAIN_JOB - 0.01, len(rows) - 0.55, "essay: ~90%\none main job", ha="right", va="center", fontsize=8,
            color="#52514e")
    ax.set_yticks(y, [names[k] for k in rows], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("share called one job (v7 rubric) or one main job (E3); mean over sizes, 95% interval", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    summary = json.loads((RESULTS_DIR / "classification_v21.json").read_text())["summary"]
    rows_a = {k: summary["arm_a"][k] for k in ARM_A_ROWS if k in summary["arm_a"]}
    one_shot = summary["investigation_same_units"][f"{ARM_B} minus {OPUS_E3}"]["one shot"]
    rows_b = {V14_AGENT: summary["arm_b"][V14_AGENT], ARM_B: summary["arm_b"][ARM_B], "one shot": {"one_main_job": one_shot}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    panel(axes[0], rows_a, ARM_A_ROWS, "A: one shot, v7's 600 real neurons and 300 pools (six sizes)")
    panel(axes[1], rows_b, ARM_B_ROWS, "B: investigating agent, v8's 120 real neurons and 120 pools (three sizes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v21_opus_essay.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
