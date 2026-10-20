"""v7 figure: each reader's flag rate by model size, for real neurons and for each kind of pooled unit."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v7 import GROUPS, READERS
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v5 import PARAMETERS
from recgen.rubric_v5 import MODELS

TITLES = {"minilm": "MiniLM (v5)", "network": "network's own representation (v6)", "llm": "language model (Haiku)"}
STYLES = {"real": ("black", "-", 2.2), "pooled 50/50": ("tab:red", "-", 1.5), "pooled 70/30": ("tab:orange", "-", 1.2),
          "pooled 90/10": ("tab:olive", "-", 1.0), "related": ("tab:blue", "--", 1.5)}


def main() -> None:
    stats = json.loads((RESULTS_DIR / "classification_v7.json").read_text())["models"]
    fig, axes = plt.subplots(1, len(READERS), figsize=(14, 4), sharey=True)
    x = np.array([PARAMETERS[m] for m in MODELS])
    for ax, reader in zip(axes, READERS):
        for g in GROUPS:
            rates = [s["rates"][reader][g] for s in stats]
            share = np.array([r["share"] for r in rates])
            low, high = np.array([r["interval"] for r in rates]).T
            color, style, width = STYLES[g]
            ax.errorbar(x, share, yerr=[share - low, high - share], color=color, linestyle=style, linewidth=width,
                        marker="o", markersize=3, capsize=2, label=g)
        ax.set_xscale("log")
        ax.set_xticks(x, MODELS)
        ax.minorticks_off()
        ax.set_title(TITLES[reader], fontsize=10)
        ax.set_xlabel("Pythia model")
    axes[0].set_ylabel("share flagged unrelated mixed")
    axes[0].legend(fontsize=8)
    fig.suptitle("Three readers on the same units (middle layer): real neurons, pools of two random neurons "
                 "at three mixes, and pools of two related neurons", fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v7_readers.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
