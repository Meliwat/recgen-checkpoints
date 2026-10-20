"""v11 figures: each reader's flag rates by model size at all three depths (the middle from v7), and
each reader's power (50/50 pools over real neurons) by depth and size."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v7 import GROUPS, READERS
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v5 import PARAMETERS
from recgen.plots_v7 import STYLES, TITLES
from recgen.rubric_v5 import MODELS
from recgen.rubric_v11 import ALL_DEPTHS

DEPTH_COLORS = {"quarter": "tab:green", "middle": "tab:gray", "three quarters": "tab:purple"}


def readers_figure(stats: dict) -> None:
    fig, axes = plt.subplots(len(ALL_DEPTHS), len(READERS), figsize=(14, 10), sharey=True, sharex=True)
    x = np.array([PARAMETERS[m] for m in MODELS])
    for row, depth in zip(axes, ALL_DEPTHS):
        for ax, reader in zip(row, READERS):
            for g in GROUPS:
                rates = [s["rates"][reader][g] for s in stats[depth]]
                share = np.array([r["share"] for r in rates])
                low, high = np.array([r["interval"] for r in rates]).T
                color, style, width = STYLES[g]
                ax.errorbar(x, share, yerr=[share - low, high - share], color=color, linestyle=style, linewidth=width,
                            marker="o", markersize=3, capsize=2, label=g)
            ax.set_xscale("log")
            ax.set_xticks(x, MODELS)
            ax.minorticks_off()
            ax.set_title(f"{TITLES[reader]}, {depth}" + (" (v7)" if depth == "middle" else ""), fontsize=9)
        row[0].set_ylabel("share flagged unrelated mixed")
    for ax in axes[-1]:
        ax.set_xlabel("Pythia model")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Three readers on the same kinds of units at three depths: real neurons, pools of two random "
                 "neurons at three mixes, and pools of two related neurons", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "v11_readers.png", dpi=110)
    plt.close(fig)


def power_figure(by_depth: dict) -> None:
    fig, axes = plt.subplots(1, 1 + len(READERS), figsize=(16, 4))
    ax = axes[0]
    for i, reader in enumerate(READERS):
        e = [by_depth[d]["excess_over_models"][reader] for d in ALL_DEPTHS]
        point = np.array([v["excess"] for v in e])
        low, high = np.array([v["interval"] for v in e]).T
        ax.errorbar(np.arange(len(ALL_DEPTHS)) + (i - 1) * 0.08, point, yerr=[point - low, high - point], marker="o",
                    capsize=3, label=TITLES[reader])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(ALL_DEPTHS)), ALL_DEPTHS)
    ax.set_ylabel("50/50 pools minus real neurons, flag rate")
    ax.set_title("Mean over the six models (95% interval)", fontsize=9)
    ax.legend(fontsize=7)
    x = np.array([PARAMETERS[m] for m in MODELS])
    for ax, reader in zip(axes[1:], READERS):
        for depth in ALL_DEPTHS:
            ax.plot(x, by_depth[depth]["excess_per_model"][reader], marker="o", color=DEPTH_COLORS[depth], label=depth)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(x, MODELS)
        ax.minorticks_off()
        ax.set_ylim(-0.2, 0.7)
        ax.set_xlabel("Pythia model")
        ax.set_title(f"{TITLES[reader]}, per model", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.suptitle("How well each reader tells 50/50 pools of two neurons from real neurons, by depth", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "v11_power.png", dpi=110)
    plt.close(fig)


def main() -> None:
    v11 = json.loads((RESULTS_DIR / "classification_v11.json").read_text())
    stats = {**v11["models"], "middle": json.loads((RESULTS_DIR / "classification_v7.json").read_text())["models"]}
    FIGURES_DIR.mkdir(exist_ok=True)
    readers_figure(stats)
    power_figure(v11["summary"]["by_depth"])


if __name__ == "__main__":
    main()
