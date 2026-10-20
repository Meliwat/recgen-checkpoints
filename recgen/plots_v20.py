"""v20 figure: flag rate and coherence by peak quartile in the top 50, both rank bands, v17's band and random
positions, per reader; and where each rank band's examples sit as a fraction of their neuron's peak."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR

PANELS = (("network, flag", "network reader: share flagged unrelated mixed", 1),
          ("network, coherence", "network reader: coherence (percentile of null)", 100),
          ("haiku, flag", "Haiku (v12's 600 fresh neurons): share flagged", 1))
STYLES = {"top": ("top 50 (ranks 1 to 50)", "-", "C0"), "band": ("ranks 201 to 500", "--", "C3"),
          "band2": ("ranks 1,001 to 2,000", "-.", "C1"), "v17": ("v17: 0.4 to 0.6 of peak", ":", "C2"),
          "reference": ("50 random positions", ":", "0.5")}
QUARTILES = ["weakest", "2nd", "3rd", "strongest"]


def main() -> None:
    results = json.loads((RESULTS_DIR / "classification_v20.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    x = [1, 2, 3, 4]
    for ax, (key, title, top) in zip(axes.flat, PANELS):
        for sample, values in results["by_quartile"][key].items():
            label, style, color = STYLES[sample]
            ax.plot(x, values["mean"], marker="o", linestyle=style, color=color, label=label)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=9)
    ax = axes.flat[3]
    ax.axhspan(0.4, 0.6, color="C2", alpha=0.15, label="v17's band")
    for offset, band in ((-0.06, "band"), (0.06, "band2")):
        label, _, color = STYLES[band]
        for i, (low, high) in enumerate(results["selectivity"][band]["median_fraction_of_peak"]):
            ax.plot([x[i] + offset] * 2, [low, high], color=color, linewidth=3, solid_capstyle="butt",
                    label=label if i == 0 else None)
    ax.set_ylim(0, 1)
    ax.set_title("median example's fraction of its neuron's peak (range over models)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    for ax in axes.flat:
        ax.set_xticks(x, QUARTILES)
        ax.set_xlabel("quartile of peak activation")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=5)
    fig.suptitle("Matched selectivity by rank: the same neurons' top 50, rank bands and v17's band "
                 "(middle layer, Pythia 70M to 2.8B)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v20_rank_band.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
