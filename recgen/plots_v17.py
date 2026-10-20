"""v17 figure: flag rate and coherence by peak quartile, top 50 against the band, per reader."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR

PANELS = (("network", "flag", "network reader: share flagged unrelated mixed"),
          ("network", "coherence", "network reader: coherence (percentile of null)"),
          ("haiku", "flag", "Haiku (598 of v12's fresh neurons): share flagged"))
STYLES = {"top": ("top 50", "-", "C0"), "band": ("band, 0.4 to 0.6 of peak", "--", "C3"),
          "reference": ("50 random positions", ":", "0.5")}


def main() -> None:
    results = json.loads((RESULTS_DIR / "classification_v17.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    x = [1, 2, 3, 4]
    for ax, (reader, measure, title) in zip(axes, PANELS):
        for sample, values in results[reader][measure]["by_quartile"].items():
            label, style, color = STYLES[sample]
            ax.plot(x, values, marker="o", linestyle=style, color=color, label=label)
        ax.set_xticks(x, ["weakest", "2nd", "3rd", "strongest"])
        ax.set_xlabel("quartile of peak activation")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 100 if measure == "coherence" else 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3)
    fig.suptitle("Below the top 50: the same neurons' top-50 and band examples (middle layer, Pythia 70M to 2.8B)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v17_band.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
