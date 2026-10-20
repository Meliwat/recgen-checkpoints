"""v22's Haiku readings: v19's units with their OpenWebText examples, shown blind with v7's prompt, system prompt and
CLI, through read_v19.read_all under v22's spend guard. Answers are cached in data/v22/llm/ (second readings in
data/v22/llm_retest/) and collected in results/v22_llm_answers.json with the cost the CLI reported.

Run with model names to read only those models' units into the cache; with none, every model's units are read or
taken from the cache and the answers file is written.
"""

import json
import sys

from recgen.config import RESULTS_DIR
from recgen.gpt2_v22 import V22_DIR, load_owt_corpus, network_path
from recgen.read_v19 import read_all
from recgen.rubric_v19 import MODELS
from recgen.rubric_v22 import SPEND_GUARD_USD
from recgen.units_v19 import build_units

LLM_DIR = V22_DIR / "llm"
RETEST_DIR = V22_DIR / "llm_retest"
ANSWERS_PATH = RESULTS_DIR / "v22_llm_answers.json"


def build_owt_units(model: str) -> list[dict]:
    """v19's units of one model, with their examples from v22's recording."""
    return build_units(model, network_path(model))


def main(names: list[str]) -> None:
    corpus = load_owt_corpus()
    answers = read_all({m: build_owt_units(m) for m in names or MODELS}, corpus, LLM_DIR, RETEST_DIR, SPEND_GUARD_USD)
    if not names:
        ANSWERS_PATH.write_text(json.dumps(answers, indent=1))
    print(f"done: ${answers['cost_usd']:.2f}; {len(answers['not_read'])} not read (spend guard), "
          f"{sum(a['label'] is None for a in answers['first'].values())} first readings without a label", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
