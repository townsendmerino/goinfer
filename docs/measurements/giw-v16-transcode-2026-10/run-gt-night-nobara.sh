#!/bin/bash
# nobara's night run for docs/tasks/task-giw-v16-transcode-2026-10.md: G-T1's real cells, G-T2, G-T2c, G-T3.
# Pre-built linux/amd64 binaries (BUILD.txt beside them names the revisions). Estimate: 85 min.
set -euo pipefail
B=$HOME/goinfer-bench/giw-v16
D=$HOME/goinfer-logs/giw-v16/$(date +%Y-%m-%d)
mkdir -p "$D"
exec python3 "$B/gt_night.py" --bin-dir "$B/bin" --out-dir "$B/out" --record "$D/gt-nobara.jsonl" --est-min 85 \
  "1.5b-cpu-amd64|$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf|int4|cpu-amd64|3" \
  "7b-cpu-amd64|$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf|int4|cpu-amd64|3" \
  "7b-cuda|$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf|int4|cuda|2" \
  "30b-dir-cpu-amd64|$HOME/models/qwen3-vl-30b-a3b-instruct|int4|cpu-amd64|2" \
  2>&1 | tee "$D/gt-nobara.log"
