#!/usr/bin/env bash
# v23 pipeline (record_v23 builds the candidates on first run). Needs v5's and v6's data/ recordings and results, v13's
# results and data/v13/'s final checkpoint. Recording is skipped if data/v23/ already holds it.
set -euo pipefail
uv run python -m recgen.record_v23
uv run python -m recgen.analyze_v23
uv run python -m recgen.plots_v23
