"""v19 figure: (a) each reader's over-all-models excess on 50/50 pools in GPT-2 beside Pythia's, and (b) each
reader's flag rate by peak quartile in GPT-2 beside Pythia's middle layer (v12)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.rubric_v19 import MINILM_MAX_EXCESS

READERS = ("minilm", "network", "llm")
NAMES = {"minilm": "MiniLM", "network": "network", "llm": "Haiku"}
COLORS = {"minilm": "#2a78d6", "network": "#eb6834", "llm": "#1baf7a"}  # categorical slots 1 to 3, validated all-pairs
MARKERS = {"minilm": "s", "network": "o", "llm": "^"}
INK, MUTED = "#0b0b0b", "#52514e"
PYTHIA_MIDDLE = {"minilm": "minilm, middle", "network": "network, middle", "llm": "haiku, middle (fresh neurons)"}


def excess_panel(ax, a: dict) -> None:
    for i, reader in enumerate(READERS):
        for family, offset, fill in (("gpt2", -0.13, COLORS[reader]), ("pythia_v7", 0.13, "white")):
            r = a[family][reader]
            low, high = r["interval"]
            x = i + offset
            ax.errorbar([x], [100 * r["excess"]], yerr=[[100 * (r["excess"] - low)], [100 * (high - r["excess"])]],
                        color=COLORS[reader], marker=MARKERS[reader], markerfacecolor=fill, markersize=8, capsize=3,
                        linewidth=2)
            ax.annotate(f"{100 * r['excess']:+.0f}", (x, 100 * high), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axhline(100 * MINILM_MAX_EXCESS, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(2.45, 100 * MINILM_MAX_EXCESS, "MiniLM's\n+10 bar", fontsize=7, color=MUTED, va="center")
    ax.plot([], [], color=MUTED, marker="o", linestyle="none", markersize=7, label="GPT-2, v19 (filled)")
    ax.plot([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="none", markersize=7,
            label="Pythia, v7's units (open)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xticks(range(len(READERS)), [NAMES[r] for r in READERS])
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("50/50 pools minus real neurons,\nflagged (points, mean over models)")
    ax.set_title("(a) Who sees two pooled neurons?", fontsize=10, loc="left")


def quartile_panel(ax, b: dict, minilm: dict) -> None:
    x = [1, 2, 3, 4]
    gpt2 = {"minilm": minilm, "network": b["network"], "llm": b["llm"]}
    for reader in READERS:
        ax.plot(x, [100 * v for v in b["pythia_v12"][PYTHIA_MIDDLE[reader]]["by_quartile"]], color=COLORS[reader],
                linestyle="--", linewidth=1.2, marker=MARKERS[reader], markerfacecolor="white", markersize=6)
        rates = [100 * v for v in gpt2[reader]["by_quartile"]]
        ax.plot(x, rates, color=COLORS[reader], linewidth=2, marker=MARKERS[reader], markersize=8,
                label=f"{NAMES[reader]} ({gpt2[reader]['n']:,} GPT-2 neurons)")
        ax.annotate(NAMES[reader], (4, rates[-1]), textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=8, color=INK)
    ax.plot([], [], color=MUTED, linewidth=2, label="GPT-2 (solid)")
    ax.plot([], [], color=MUTED, linewidth=1.2, linestyle="--", markerfacecolor="white", label="Pythia middle layer, v12 (dashed)")
    ax.set_xticks(x, ["weakest", "2nd", "3rd", "strongest"])
    ax.set_xlim(0.7, 4.7)
    ax.set_ylim(0, 85)  # headroom for the legend
    ax.set_xlabel("quartile of peak activation, within each model")
    ax.set_ylabel("flagged unrelated mixed (%)")
    ax.set_title("(b) Do strongly firing neurons look cleaner?", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="upper right")


def main() -> None:
    summary = json.loads((RESULTS_DIR / "classification_v19.json").read_text())["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    excess_panel(axes[0], summary["a"])
    quartile_panel(axes[1], summary["b"], summary["secondary"]["minilm_magnitude"])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("v19: the two methods findings on GPT-2 small, medium and large (middle layer), beside Pythia",
                 fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v19_gpt2.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
