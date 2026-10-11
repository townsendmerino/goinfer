#!/bin/bash
# nobara's copy-floor re-read with T8 in the new arm (docs/completed/task-giw-v16-transcode-2026-10.md, T8: recorded, not gated).
# The earlier night runs' cells and driver. old = the T4b build, new = the same with T8 (bin-t8/BUILD.txt). Estimate: 30 min.
set -euo pipefail
B=$HOME/goinfer-bench/giw-v16
D=$HOME/goinfer-logs/giw-v16/$(date +%Y-%m-%d)
mkdir -p "$D"
exec python3 "$B/gt_night.py" --bin-dir "$B/bin-t8" --out-dir "$B/out" --record "$D/gt-nobara-t8.jsonl" --est-min 30 \
  "1.5b-cpu-amd64|$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf|int4|cpu-amd64|3" \
  "7b-cpu-amd64|$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf|int4|cpu-amd64|3" \
  "7b-cuda|$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf|int4|cuda|2" \
  "30b-dir-cpu-amd64|$HOME/models/qwen3-vl-30b-a3b-instruct|int4|cpu-amd64|2" \
  2>&1 | tee "$D/gt-nobara-t8.log"
