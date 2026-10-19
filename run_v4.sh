#!/usr/bin/env bash
# v4 pipeline. Needs v1's and v2's results/ and data/corpus.npy. Recording skips checkpoints already cached in data/v4/.
set -euo pipefail
uv run python -m recgen.formation
uv run python -m recgen.analyze_v4
uv run python -m recgen.plots_v4
