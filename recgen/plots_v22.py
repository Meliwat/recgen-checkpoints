"""v22 figure: v19's GPT-2 neurons on OpenWebText beside the Pile. (a) each reader's excess on 50/50 pools, (c) Haiku's
flag rate on the same real neurons, with Pythia's at the nearest sizes, and (b) each reader's flag rate by peak
quartile."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v19 import COLORS, INK, MARKERS, MUTED, NAMES, READERS
from recgen.rubric_v19 import MINILM_MAX_EXCESS, MODELS
from recgen.rubric_v22 import PYTHIA_SIZE_MATCHED

SIZES = {"gpt2": "small", "gpt2-medium": "medium", "gpt2-large": "large"}


def errorbar(ax, x: float, value: float, interval: list[float], color: str, marker: str, fill: str) -> None:
    ax.errorbar([x], [100 * value], yerr=[[100 * (value - interval[0])], [100 * (interval[1] - value)]], color=color,
                marker=marker, markerfacecolor=fill, markersize=8, capsize=3, linewidth=2)


def corpus_legend(ax, loc: str) -> None:
    ax.plot([], [], color=MUTED, marker="o", linestyle="none", markersize=7, label="OpenWebText, v22 (filled)")
    ax.plot([], [], color=MUTED, marker="o", markerfacecolor="white", linestyle="none", markersize=7,
            label="Pile, v19 (open)")
    ax.legend(fontsize=7, loc=loc)


def excess_panel(ax, a: dict) -> None:
    for i, reader in enumerate(READERS):
        for corpus, offset, fill in (("openwebtext", -0.13, COLORS[reader]), ("pile_v19", 0.13, "white")):
            r = a[corpus][reader]
            errorbar(ax, i + offset, r["excess"], r["interval"], COLORS[reader], MARKERS[reader], fill)
            ax.annotate(f"{100 * r['excess']:+.0f}", (i + offset, 100 * r["interval"][1]), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axhline(100 * MINILM_MAX_EXCESS, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(2.45, 100 * MINILM_MAX_EXCESS, "MiniLM's\n+10 bar", fontsize=7, color=MUTED, va="center")
    corpus_legend(ax, "upper left")
    ax.set_xticks(range(len(READERS)), [NAMES[r] for r in READERS])
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylim(-10, 50)  # headroom for the legend
    ax.set_ylabel("50/50 pools minus real neurons,\nflagged (points, mean over models)")
    ax.set_title("(a) Who sees two pooled neurons?", fontsize=10, loc="left")


def haiku_panel(ax, c: dict) -> None:
    color, marker = COLORS["llm"], MARKERS["llm"]
    for i, model in enumerate(MODELS):
        r = c["by_model"][model]
        ax.plot([i - 0.13, i + 0.13], [100 * r["openwebtext"], 100 * r["pile"]], color=color, linewidth=1, alpha=0.5)
        ax.plot([i - 0.13], [100 * r["openwebtext"]], color=color, marker=marker, markersize=8, linestyle="none")
        ax.plot([i + 0.13], [100 * r["pile"]], color=color, marker=marker, markerfacecolor="white", markersize=8,
                linestyle="none")
        ax.annotate(f"{100 * r['openwebtext']:.0f}%", (i - 0.13, 100 * r["openwebtext"]), textcoords="offset points",
                    xytext=(-6, 0), ha="right", va="center", fontsize=8, color=INK)
    ax.axhline(100 * c["pythia_size_matched"], color=MUTED, linewidth=1, linestyle="--")
    ax.text(2.45, 100 * c["pythia_size_matched"], f"Pythia {', '.join(PYTHIA_SIZE_MATCHED)}\n(v7, mean)",
            fontsize=7, color=MUTED, va="center")
    corpus_legend(ax, "upper left")
    ax.set_xticks(range(len(MODELS)), [f"GPT-2 {SIZES[m]}" for m in MODELS])
    ax.set_xlim(-0.6, 3.5)
    ax.set_ylim(0, 70)
    ax.set_ylabel("real neurons flagged unrelated mixed (%)")
    ax.set_title(f"(c) Haiku on the same real neurons: {100 * c['change']:+.0f} points", fontsize=10, loc="left")


def quartile_panel(ax, b: dict, minilm: dict) -> None:
    x = [1, 2, 3, 4]
    owt = {"minilm": minilm, "network": b["network"], "llm": b["llm"]}
    for reader in ("network", "llm"):
        ax.plot(x, [100 * v for v in b["pile_v19"][reader]["by_quartile"]], color=COLORS[reader], linestyle="--",
                linewidth=1.2, marker=MARKERS[reader], markerfacecolor="white", markersize=6)
    for reader in READERS:
        rates = [100 * v for v in owt[reader]["by_quartile"]]
        ax.plot(x, rates, color=COLORS[reader], linewidth=2, marker=MARKERS[reader], markersize=8)
        ax.annotate(NAMES[reader], (4, rates[-1]), textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=8, color=INK)
    ax.plot([], [], color=MUTED, linewidth=2, label="OpenWebText, v22 (solid)")
    ax.plot([], [], color=MUTED, linewidth=1.2, linestyle="--", label="Pile, v19 (dashed)")
    ax.set_xticks(x, ["weakest", "2nd", "3rd", "strongest"])
    ax.set_xlim(0.7, 4.8)
    ax.set_ylim(0, 85)
    ax.set_xlabel("quartile of peak activation, within each model")
    ax.set_ylabel("flagged unrelated mixed (%)")
    ax.set_title("(b) Do strongly firing neurons look cleaner?", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="upper right")


def main() -> None:
    summary = json.loads((RESULTS_DIR / "classification_v22.json").read_text())["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    excess_panel(axes[0], summary["a"])
    haiku_panel(axes[1], summary["c"])
    quartile_panel(axes[2], summary["b"], summary["secondary"]["minilm_magnitude"])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("v22: v19's GPT-2 neurons (middle layer) read on OpenWebText, beside the same neurons on the Pile",
                 fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v22_openwebtext.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
