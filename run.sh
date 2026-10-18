#!/usr/bin/env bash
# Full pipeline. Selection reads only the final checkpoint; tracing then records every checkpoint.
set -euo pipefail
uv run python -m recgen.record --steps 143000
uv run python -m recgen.select
uv run python -m recgen.record
uv run python -m recgen.analyze
uv run python -m recgen.plots
