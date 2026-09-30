#!/bin/bash
# D6b (docs/tasks/task-constrained-confidence.md): goinfer's Route B on autotrust's JEV-9B over the 150 D0 items, one arm
# per invocation (ARM = f32 | int8int8 | int4), CPU, graded afterwards by grade.py against the reference. Unattended:
# BIN is a goinfer-chat pre-built at 90f8dbc9 (the batched Qwen3.5 PromptHidden), the items and the model are read from
# fixed paths, output goes to a durable results dir.
#   ARM=int4 bash run-d6b.sh
# The 2026-09-30 amendment adds a GPU arm: ARM=int4-cuda QUANT=int4 BACKEND=cuda BIN=<a build at f7e1782a or later>. QUANT
# defaults to ARM and BACKEND to cpu, so the three CPU arms run exactly as first queued.
set -euo pipefail
ARM=${ARM:?f32, int8int8, int4 or int4-cuda}
QUANT=${QUANT:-$ARM}
BACKEND=${BACKEND:-cpu}
B=${B:-$HOME/goinfer-bench/decisions-d6b-2026-09}
BIN=${BIN:-$B/goinfer-chat-cuda-90f8dbc9}
ITEMS=${ITEMS:-$HOME/mycode/goinfer/testdata/decisions/items.jsonl}
M=$HOME/models/JEV-9B
mkdir -p "$B/results"
[ -x "$BIN" ] || { echo "no binary at $BIN" >&2; exit 1; }
echo "[$(date '+%H:%M:%S')] D6b arm $ARM: $BIN --backend $BACKEND --quant $QUANT on $(wc -l < "$ITEMS") items"
"$BIN" decide --model "$M" --head "$M" --backend "$BACKEND" --quant "$QUANT" -o "$B/results/$ARM.jsonl" "$ITEMS" 2> "$B/results/$ARM.stderr"
echo "[$(date '+%H:%M:%S')] done: $(wc -l < "$B/results/$ARM.jsonl") rows; stderr tail:"
tail -3 "$B/results/$ARM.stderr"
