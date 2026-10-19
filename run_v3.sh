#!/usr/bin/env bash
# v3 pipeline. Needs v1's results/ and data/ (corpus, final-step window maxima and neuron activations)
# and v2's results/classification_v2.json. The recording step skips work already cached in data/.
set -euo pipefail
uv run python -m recgen.facets_v3   # enlarge v1's facets (input side, sentence model)
uv run python -m recgen.outputs     # final checkpoint: ablations and patches (skipped if cached)
uv run python -m recgen.analyze_v3
uv run python -m recgen.plots_v3
