"""v6 figures: the share of mixed-looking neurons by model size under the network-native detector
and v5's MiniLM detector, and the detector checks."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v5 import PARAMETERS
from recgen.rubric_v5 import DEPTHS, MODELS, PRIMARY_DEPTH

MEASURES = {"unrelated_mixed": "unrelated mixed", "multi_facet": "2+ facets of any kind"}
DETECTORS = {"v6": ("network-native (v6)", "-"), "v5": ("MiniLM (v5)", "--")}


def rows(layers: list[dict], depth: str) -> list[dict]:
    return [next(s for s in layers if s["model"] == m and s["depth"] == depth) for m in MODELS]


def errorbar(ax, x, share, interval, **kwargs):
    share, (low, high) = np.array(share), np.array(interval).T
    ax.errorbar(x, share, yerr=[share - low, high - share], marker="o", capsize=3, **kwargs)


def size_axis(ax, x):
    ax.set_xscale("log")
    ax.set_xticks(x, MODELS)
    ax.minorticks_off()


def scale_figure(v6: list[dict], v5: list[dict], x: np.ndarray) -> None:
    fig, axes = plt.subplots(len(MEASURES), len(DEPTHS), figsize=(12, 6.5), sharey="row")
    for row, (measure, title) in zip(axes, MEASURES.items()):
        for ax, depth in zip(row, DEPTHS):
            ours = rows(v6, depth)
            errorbar(ax, x, [r[measure]["share"] for r in ours], [r[measure]["interval"] for r in ours],
                     color="C0", linestyle=DETECTORS["v6"][1], label=DETECTORS["v6"][0])
            theirs = rows(v5, depth)
            errorbar(ax, x, [r[f"{measure}_share"] for r in theirs], [r[f"{measure}_interval"] for r in theirs],
                     color="C1", linestyle=DETECTORS["v5"][1], label=DETECTORS["v5"][0])
            size_axis(ax, x)
            ax.set_ylim(0, 1)
            ax.set_title(f"{title}, {depth} depth" + (" (primary)" if depth == PRIMARY_DEPTH and measure == "unrelated_mixed" else ""),
                         fontsize=9)
        row[0].set_ylabel("share of sampled neurons")
    axes[0][0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Pythia model")
    fig.suptitle("Mixed-looking neurons by model size, two detectors on the same neurons (95% bootstrap intervals)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "v6_scale.png", dpi=110)
    plt.close(fig)


def checks_figure(v6: list[dict], x: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for detector, (label, style) in DETECTORS.items():
        for depth in DEPTHS:
            ours = rows(v6, depth)
            width = 2.2 if depth == PRIMARY_DEPTH else 1
            color = f"C{list(DEPTHS).index(depth)}"
            power = [r["checks"]["mixture_power"][detector] for r in ours]
            errorbar(axes[0], x, [p["share"] for p in power], [p["interval"] for p in power], color=color,
                     linestyle=style, linewidth=width, label=f"{label}, {depth}")
            axes[1].plot(x, [r["checks"]["no_facet_share"][detector] for r in ours], marker="o", color=color,
                         linestyle=style, linewidth=width)
    for depth in DEPTHS:
        axes[2].plot(x, [r["agreement"]["kappa"] for r in rows(v6, depth)], marker="o", color=f"C{list(DEPTHS).index(depth)}",
                     linewidth=2.2 if depth == PRIMARY_DEPTH else 1, label=f"{depth} depth")
    titles = ["unrelated mixed among two-neuron mixtures (power)", "share of neurons with no facet",
              "agreement of the two detectors (Cohen's kappa)"]
    for ax, title in zip(axes, titles):
        size_axis(ax, x)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Pythia model")
    axes[0].set_ylim(0, 1.25)
    axes[1].set_ylim(0, None)
    axes[0].legend(fontsize=6, ncol=2, loc="upper center")
    axes[2].legend(fontsize=7)
    fig.suptitle("Detector checks. Left and middle: solid lines are the network-native detector, dashed lines MiniLM",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "v6_checks.png", dpi=110)
    plt.close(fig)


def main() -> None:
    v6 = json.loads((RESULTS_DIR / "classification_v6.json").read_text())["layers"]
    v5 = json.loads((RESULTS_DIR / "classification_v5.json").read_text())["layers"]
    x = np.array([PARAMETERS[m] for m in MODELS])
    FIGURES_DIR.mkdir(exist_ok=True)
    scale_figure(v6, v5, x)
    checks_figure(v6, x)


if __name__ == "__main__":
    main()
