"""v5 figure: share of mixed-looking neurons by model size, at each depth, with 95% bootstrap intervals."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.rubric_v5 import DEPTHS, MODELS, PRIMARY_DEPTH

PARAMETERS = {"70m": 70e6, "160m": 160e6, "410m": 410e6, "1b": 1.0e9, "1.4b": 1.4e9, "2.8b": 2.8e9}
MEASURES = {"unrelated_mixed": "unrelated mixed (primary)", "multi_facet": "2+ facets of any kind"}


def main() -> None:
    stats = json.loads((RESULTS_DIR / "classification_v5.json").read_text())["layers"]
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(11, 3.8), sharey=True)
    x = np.array([PARAMETERS[m] for m in MODELS])
    for ax, (measure, title) in zip(axes, MEASURES.items()):
        for depth in DEPTHS:
            rows = [next(s for s in stats if s["model"] == m and s["depth"] == depth) for m in MODELS]
            share = np.array([r[f"{measure}_share"] for r in rows])
            low, high = np.array([r[f"{measure}_interval"] for r in rows]).T
            ax.errorbar(x, share, yerr=[share - low, high - share], marker="o", capsize=3,
                        linewidth=2.2 if depth == PRIMARY_DEPTH else 1, label=f"{depth} depth")
        ax.set_xscale("log")
        ax.set_xticks(x, MODELS)
        ax.minorticks_off()
        ax.set_xlabel("Pythia model")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("share of sampled neurons")
    axes[0].legend(fontsize=8)
    fig.suptitle("Mixed-looking neurons by model size (v1's facet detector, 95% bootstrap intervals)", fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v5_scale.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
