#!/bin/bash
# D7 (docs/measurements/decisions-d7-2026-09-28.md section 4, pre-registered 2026-09-28; harness built and frozen 2026-10-02): decisions against schema-constrained
# generation and a forced tool call, speed, on nobara's CUDA, Qwen3.5-9B Q4_K_M, batch 1. A night job (a timed measurement): the harness idle-gates each state block at
# BENCH_MAX_LOADAVG (default 1.0, the registered value; a raised value voids the run) and the night runner holds the timing lock for the job.
# Unattended: a PINNED serve-cuda, a PINNED copy of the harness and the FROZEN prompt file, each sha256-checked; output to a durable directory; no prompts.
# Rows expected: 3 K x 8 states x (3 single-question arms + 2 five-question arms) + 8 states at K=32 (decision x5 only) = 128. Estimate 60 min.
set -euo pipefail
B=${B:-$HOME/goinfer-bench/decisions-d7}
SERVE=$B/serve-cuda-644d8008;  SERVE_SHA=365425d917281c6e90c7723891ad28276611bd7e0f84366dafc2d9461b014ee9
PROMPTS=$B/prompts.json;       PROMPTS_SHA=2d56d87f800e2b7235bbeb9e1148039daee1e3d22c53a503ea26711ba0910dd9
HARNESS=$B/d7_bench.py;        HARNESS_SHA=643762c48b336f933d491f554b352ac6d1e24e03408ddf83c90c62c6577a0ca4
MODEL=$HOME/models/Qwen3.5-9B-Q4_K_M.gguf
OUT=$B/run-$(date +%F)
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "$1: sha256 differs from the registered one"; exit 1; }; }
chk "$SERVE" "$SERVE_SHA"; chk "$PROMPTS" "$PROMPTS_SHA"; chk "$HARNESS" "$HARNESS_SHA"
case "$MODEL" in /srv/models/*|/Volumes/*) echo "model is on the archive, not the bench set"; exit 1;; esac
[ -f "$MODEL" ] || { echo "no model at $MODEL"; exit 1; }
mkdir -p "$OUT"; RAW=$OUT/raw.jsonl
[ ! -e "$RAW" ] || { echo "$RAW exists: a re-run must not mix rows"; exit 1; }
echo "$(date '+%H:%M:%S') == D7 start; host $(uname -srm); load $(cut -d' ' -f1-3 /proc/loadavg); serve $(basename "$SERVE")"
GOINFER_SERVE_BIN="$SERVE" D7_MODEL="$MODEL" python3 -u "$HARNESS" run --prompts "$PROMPTS" --raw "$RAW" --log "$OUT/servers.log"
n=$(grep -c '"arm"' "$RAW" || true)
echo "$(date '+%H:%M:%S') == run done; rows: $n"
python3 "$HARNESS" analyze --raw "$RAW" | tee "$OUT/analysis.txt"
# An unattended run must not report success without its output.
[ "$n" -eq 128 ] || { echo "expected 128 rows, got $n"; exit 1; }
echo "$(date '+%H:%M:%S') == DONE; results in $OUT"
