#!/usr/bin/env bash
# The sampled 35B load (README.md in this directory). Pinned binary, no prompts; about 4 minutes. night.py holds the timing lock.
#   python3 scripts/night.py add c3-35b-sampled-load --est 8 --by "<who>" --doc docs/measurements/c3-35b-sampled-load-2026-10-19/README.md -- bash docs/measurements/c3-35b-sampled-load-2026-10-19/run-sampled-load.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HOME/serve-cuda-333407c3}
OUT=${1:-$HOME/goinfer-logs/c3-35b-sampled-load-$(date +%F)}
MODEL=$HOME/models/qwen3.6-35b-a3b-q4_k_m.gguf
[ -x "$BIN" ] || { echo "FATAL: $BIN is missing"; exit 2; }
[ -f "$MODEL" ] && [ -f "${MODEL%.gguf}.int4.e4h.cuda.v17.giw" ] || { echo "FATAL: the model or its sidecar is missing"; exit 2; }
mkdir -p "$OUT"
echo "binary $BIN, started $(date '+%F %T %Z')"
python3 "$HERE/sample35b.py" --serve "$BIN" --model "$MODEL" --out "$OUT" 2>&1 | tee "$OUT/report.txt"
rc=${PIPESTATUS[0]}
[ -f "$OUT/report.json" ] || { echo "!! no report: the run produced no samples"; exit 1; }
echo "finished $(date '+%F %T %Z') rc=$rc"
exit $rc
