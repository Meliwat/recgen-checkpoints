"""v9 figure: each reader's flag rate for natural unions by how many examples come from the minority
neuron (pooled over models), against its rate on real neurons."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v9 import READERS
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v7 import TITLES


def main() -> None:
    tests = json.loads((RESULTS_DIR / "classification_v9.json").read_text())["summary"]["tests"]
    fig, axes = plt.subplots(1, len(READERS), figsize=(13, 3.8), sharey=True)
    for ax, reader in zip(axes, READERS):
        bins = tests[reader]["by_minority"]["bins"]
        x = np.arange(len(bins))
        ax.bar(x, [b["share"] for b in bins], color="tab:red", alpha=0.7, label="natural unions")
        for i, b in enumerate(bins):
            ax.text(i, b["share"] + 0.01, f"n={b['n']}", ha="center", fontsize=7)
        ax.axhline(tests[reader]["real_rate"], color="black", linewidth=1.5, label="real neurons")
        ax.set_xticks(x, [b["minority"] for b in bins])
        ax.set_xlabel("examples from the minority neuron (of 50)")
        ax.set_title(TITLES[reader], fontsize=10)
    axes[0].set_ylabel("share flagged unrelated mixed")
    axes[0].legend(fontsize=8)
    fig.suptitle("Natural unions of two neurons, pooled over Pythia 70M to 2.8B (middle layer)", fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v9_unions.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
