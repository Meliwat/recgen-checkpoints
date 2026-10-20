"""v14 figure: v8's figure (recgen/plots_v8.draw) with the Opus agent as a sixth reader."""

import json

from recgen.analyze_v14 import AGENT_NAME, LLM_READERS, READERS
from recgen.config import RESULTS_DIR
from recgen.plots import FIGURES_DIR
from recgen.plots_v8 import NAMES, draw


def main() -> None:
    summary = json.loads((RESULTS_DIR / "classification_v14.json").read_text())["summary"]
    draw(summary, READERS, LLM_READERS, {**NAMES, AGENT_NAME: "Opus agent (v14)"}, FIGURES_DIR / "v14_readers.png",
         height=4.8)


if __name__ == "__main__":
    main()
