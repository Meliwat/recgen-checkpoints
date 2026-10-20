"""v12 figure: flag rate by peak-activation quartile, for each held-out test and the discovery layer."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR


def main() -> None:
    results = json.loads((RESULTS_DIR / "classification_v12.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 4))
    x = [1, 2, 3, 4]
    for part, style in (("held_out", "-"), ("discovery", ":")):
        for key, r in results[part].items():
            ax.plot(x, r["by_quartile"], marker="o", linestyle=style, label=f"{key} ({part.replace('_', '-')})")
    ax.set_xticks(x, ["weakest", "2nd", "3rd", "strongest"])
    ax.set_xlabel("quartile of peak activation")
    ax.set_ylabel("share flagged unrelated mixed")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Do strongly firing neurons look cleaner? Held-out tests (solid) and the discovery layer (dotted)", fontsize=9)
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "v12_magnitude.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
