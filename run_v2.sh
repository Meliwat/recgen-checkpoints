#!/usr/bin/env bash
# v2 pipeline. Needs v1's results/ and data/corpus.npy. Each recording step skips work already cached in data/.
set -euo pipefail
uv run python -m recgen.attribution          # final checkpoint: upstream attributions (skipped if cached)
uv run python -m recgen.bridges select       # pool embeddings, bridge and control sets
uv run python -m recgen.bridges record       # pre-activations at the dense checkpoints
uv run python -m recgen.analyze_v2
uv run python -m recgen.plots_v2
