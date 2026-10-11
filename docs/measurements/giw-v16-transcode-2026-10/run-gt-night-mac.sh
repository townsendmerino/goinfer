#!/bin/bash
# The Mac's night run for docs/completed/task-giw-v16-transcode-2026-10.md: G-T1's real cells, G-T2, G-T2c, G-T3.
# Pre-built binaries (BUILD.txt beside them names the revisions). Estimate: 20 min.
set -euo pipefail
B=$HOME/goinfer-bench/giw-v16
D=$HOME/goinfer-logs/giw-v16/$(date +%Y-%m-%d)
mkdir -p "$D"
T=${GT_TAG:+-$GT_TAG} # a second run on one date keeps its own record: GT_TAG=t4b writes gt-<box>-t4b.jsonl
exec python3 "$B/gt_night.py" --bin-dir "$B/bin" --out-dir "$B/out" --record "$D/gt-mac$T.jsonl" --est-min 20 \
  "1.5b-cpu-arm64|$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf|int4|cpu-arm64|3" \
  "1.5b-metal|$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf|int4|metal|3" \
  "7b-metal|$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf|int4|metal|2" \
  2>&1 | tee "$D/gt-mac$T.log"
