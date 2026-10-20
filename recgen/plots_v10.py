"""v10 figure: each language-model reader's flag rates by model size (as v7's figure), and every
reader's over-all-models 50/50 excess with its interval."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recgen.analyze_v7 import GROUPS
from recgen.analyze_v10 import ALL_READERS, LLM_READERS
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v5 import PARAMETERS
from recgen.plots_v7 import STYLES
from recgen.rubric_v5 import MODELS

TITLES = {"haiku": "Haiku (v7)", "sonnet": "Sonnet", "opus": "Opus"}


def main() -> None:
    summary = json.loads((RESULTS_DIR / "classification_v10.json").read_text())["summary"]
    fig, axes = plt.subplots(1, len(LLM_READERS) + 1, figsize=(17, 4),
                             gridspec_kw={"width_ratios": [1] * len(LLM_READERS) + [0.8]})
    x = np.array([PARAMETERS[m] for m in MODELS])
    for ax, reader in zip(axes, LLM_READERS):
        tests = summary["v7_tests"][reader]
        if tests is not None:
            for g in GROUPS:
                rates = [tests["per_model"][m]["rates"][g] for m in MODELS]
                share = np.array([r["share"] for r in rates])
                low, high = np.array([r["interval"] for r in rates]).T
                color, style, width = STYLES[g]
                ax.errorbar(x, share, yerr=[share - low, high - share], color=color, linestyle=style, linewidth=width,
                            marker="o", markersize=3, capsize=2, label=g)
        ax.set_xscale("log")
        ax.set_xticks(x, MODELS)
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_title(TITLES[reader], fontsize=10)
        ax.set_xlabel("Pythia model")
    axes[0].set_ylabel("share flagged unrelated mixed")
    axes[0].legend(fontsize=8)
    ax = axes[-1]
    for i, reader in enumerate(ALL_READERS):
        e = summary["excess"][reader]
        ax.errorbar(i, e["excess"], yerr=[[e["excess"] - e["interval"][0]], [e["interval"][1] - e["excess"]]],
                    marker="o", capsize=3, color="tab:red" if reader in LLM_READERS else "gray")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(ALL_READERS)), ALL_READERS)
    ax.set_title("50/50 pools minus real neurons,\nmean over models (95% interval)", fontsize=10)
    fig.suptitle("Stronger language-model readers on v7's units (middle layer): real neurons, pools of two random "
                 "neurons at three mixes, and pools of two related neurons", fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v10_readers.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
