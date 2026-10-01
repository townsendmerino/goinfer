#!/usr/bin/env bash
# B9's split, one arm per job (grade.py here holds the registered reading). Route A, bare-v1, raw, --ctx 4096, on
# Qwen3.5-9B-Q4_K_M.gguf, CPU, at ARM's quant, over the 100 rows of noul-100-ids.txt. That is D6a's arm B with only the
# backend and quant changed. Route A on the CPU runs per token (~0.19 s per prompt token, probed 2026-10-01: 101 s and
# 23 s for 516- and 124-token rows), so 100 rows are ~1.5 h per arm. nobara-pc, ~/models only.
#
#   ARM=int4 bash run-b9.sh        ARM=int8int8 bash run-b9.sh
set -euo pipefail
: "${ARM:?int4 or int8int8}"
B=$HOME/goinfer-bench/b9-split-2026-10-01
BIN=$B/goinfer-chat-cpu-4fa285ab   # built from demo/chat at 4fa285ab (main, aikit v1.51.1)
M=$HOME/models/Qwen3.5-9B-Q4_K_M.gguf
LOG=$HOME/goinfer-logs/b9-route-a-cpu-2026-10-01
mkdir -p "$B/results" "$LOG"
case "$M" in /srv/models/*|/Volumes/*) echo "model $M is on the archive, not the bench set"; exit 1;; esac
[ -x "$BIN" ] && [ -f "$M" ] && [ -f "$B/noul-400.jsonl" ] && [ -f "$B/noul-100-ids.txt" ]
python3 - "$B" <<'PY'
import json, sys
B = sys.argv[1]
ids = [l.strip() for l in open(B + "/noul-100-ids.txt") if l.strip()]
rows = {json.loads(l)["id"]: l for l in open(B + "/noul-400.jsonl") if l.strip()}
open(B + "/noul-100.jsonl", "w").write("".join(rows[i] if rows[i].endswith("\n") else rows[i] + "\n" for i in ids))
print("rows", len(ids))
PY
echo "[$(date '+%F %T %Z')] arm cpu-$ARM: $(basename "$BIN"), $($BIN --version 2>&1 | head -1)" | tee -a "$LOG/run.log"
"$BIN" decide --model "$M" --backend cpu --quant "$ARM" --ctx 4096 --template bare-v1 \
  -o "$B/results/cpu-$ARM.jsonl" "$B/noul-100.jsonl" 2> "$B/results/cpu-$ARM.stderr"
echo "[$(date '+%F %T %Z')] arm cpu-$ARM done: $(wc -l < "$B/results/cpu-$ARM.jsonl") rows; $(tail -1 "$B/results/cpu-$ARM.stderr")" | tee -a "$LOG/run.log"
