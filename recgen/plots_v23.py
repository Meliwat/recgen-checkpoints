"""v23 figure: driver relatedness of each neuron's least-overlapping facet pair, by peak."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR

QUARTILE_NAMES = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]


def rate_with_error(values: list[bool], z: float = 1.96) -> tuple[float, list[float]]:
    """Share and its Wilson 95% interval as distances below and above the share (descriptive)."""
    n, p = len(values), float(np.mean(values))
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, [max(p - (centre - half), 0.0), max(centre + half - p, 0.0)]  # clip rounding below 0 at p = 0 or 1


def plot(rows: list[dict], path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flat
    x = np.arange(4)
    series = {"least pair driver-unrelated (MLP drivers)": lambda r: r["mlp"]["least_unrelated"],
              "least pair driver-unrelated (head drivers)": lambda r: r["heads"]["least_unrelated"],
              "reader's flag: unrelated mixed (v6)": lambda r: r["unrelated_mixed"]}
    width = 0.8 / len(series)
    for i, (name, get) in enumerate(series.items()):
        rates = [rate_with_error([get(r) for r in rows if r["quartile"] == k]) for k in range(4)]
        axes[0].bar(x + (i - 1) * width, [p for p, _ in rates], width, yerr=np.array([e for _, e in rates]).T, capsize=3, label=name)
    axes[0].set_xticks(x, QUARTILE_NAMES, fontsize=8)
    axes[0].set_ylim(0, 1.45)
    axes[0].set_ylabel("share of neurons")
    axes[0].set_title("Pre-registered: by peak quartile (50 neurons each, Wilson 95% bars)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")

    for k in range(4):
        group = [r for r in rows if r["quartile"] == k]
        axes[1].scatter([r["peak"] for r in group], [r["mlp"]["margin"] for r in group], s=12, alpha=0.7, label=QUARTILE_NAMES[k])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xticks([2, 2.5, 3, 4, 5])
    axes[1].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("peak activation (log scale)")
    axes[1].set_ylabel("least pair's overlap minus null threshold")
    axes[1].set_title("Margin of the least-overlapping pair (MLP drivers)\nabove 0: shares drivers beyond the null", fontsize=10)
    axes[1].legend(fontsize=7)

    thresholds = [[r["mlp"]["threshold"] for r in rows if r["quartile"] == k] for k in range(4)]
    least = [[r["mlp"]["least_overlap"] for r in rows if r["quartile"] == k] for k in range(4)]
    axes[2].boxplot(thresholds, positions=x - 0.18, widths=0.3, patch_artist=True, boxprops={"facecolor": "#cfd8e3"})
    axes[2].boxplot(least, positions=x + 0.18, widths=0.3, patch_artist=True, boxprops={"facecolor": "#f2d0b5"})
    for k in range(4):
        sizes = [s for r in rows if r["quartile"] == k for s in r["facet_sizes"]]
        axes[2].text(k, -0.04, f"facet size {np.median(sizes):g}", ha="center", fontsize=7)
    axes[2].set_xticks(x, QUARTILE_NAMES, fontsize=8)
    axes[2].set_ylabel("top-100 MLP driver overlap")
    axes[2].set_ylim(-0.06, None)
    axes[2].set_title("Null threshold (blue) and least pair's overlap (orange), MLP drivers", fontsize=10)

    for i, (name, flag) in enumerate((("reader flags unrelated mixed", True), ("reader doesn't flag", False))):
        rates = [rate_with_error([r["mlp"]["least_unrelated"] for r in rows if r["quartile"] == k and r["unrelated_mixed"] == flag])
                 for k in range(4)]
        counts = [sum(r["quartile"] == k and r["unrelated_mixed"] == flag for r in rows) for k in range(4)]
        axes[3].bar(x + (i - 0.5) * 0.4, [p for p, _ in rates], 0.4, yerr=np.array([e for _, e in rates]).T, capsize=3,
                    label=f"{name} (n = {', '.join(map(str, counts))})")
    axes[3].set_xticks(x, QUARTILE_NAMES, fontsize=8)
    axes[3].set_ylim(0, 1.3)
    axes[3].set_ylabel("least pair driver-unrelated (MLP drivers)")
    axes[3].set_title("Exploratory, after the results: split by the reader's own flag", fontsize=10)
    axes[3].legend(fontsize=8, loc="upper right")

    fig.suptitle("v23: are weakly firing neurons' facets driven by different upstream circuits? "
                 "(Pythia-410m layer 12, final checkpoint)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    data = json.loads((RESULTS_DIR / "classification_v23.json").read_text())
    FIGURES_DIR.mkdir(exist_ok=True)
    plot(data["neurons"], FIGURES_DIR / "v23_drivers_by_peak.png")


if __name__ == "__main__":
    main()
