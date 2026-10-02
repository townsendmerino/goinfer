#!/bin/bash
# D6a, the owed measurement (decisions-d6a-2026-09-28.md, "Owed, by night"; pre-registered there, 2026-10-02, before this ran): goinfer's Route A
# (bare-v1, base Qwen3.5-9B Q4_K_M, no head, no calibration) at --quant q4k on the CPU, over the 150 D0 items, the same measurement as
# results/b0cmp-goinfer-bare.jsonl (int4, CUDA) with the quantization the only intended change. A record, not a gate: nothing here changes D6a's
# BUILD D2-D4 verdict. Unattended: a PINNED goinfer-chat (the D6a-era binary the int4 arm's neighbours used), a PINNED copy of the 150 items
# (sha256 checked), models from ~/models, output to a durable directory, no prompts. ~27,861 prompt tokens at the first attempt's measured 290 ms
# per prompt token on the CPU (rows 1-6, 2026-09-30) = ~2.25 h; queued as 150 min. Output is written row by row, so progress is visible in $OUT.
set -euo pipefail
B=${B:-$HOME/goinfer-bench/decisions-d6a}
BIN=${BIN:-$B/goinfer-chat-cuda-a4e16c43}
ITEMS=$B/d6a-q4k-items-150.jsonl
OUT=$B/b0cmp-goinfer-bare-q4k.jsonl
M9=$HOME/models/Qwen3.5-9B-Q4_K_M.gguf
BIN_SHA=c9974a122c8755fca901f53b2deec0fdb8b7fe85f8ec1f666c938b2105d4ee5d
ITEMS_SHA=2a5f37e57d51a0b0139b70381c98ce19053aff0a8f0935f50a9452f1382da7fc
ts() { date '+%H:%M:%S'; }
case "$M9" in /srv/models/*|/Volumes/*) echo "model $M9 is on the archive, not the bench set"; exit 1;; esac
[ -f "$M9" ] || { echo "no model at $M9"; exit 1; }
[ -x "$BIN" ] || { echo "no binary $BIN"; exit 1; }
[ "$(sha256sum "$BIN" | cut -d' ' -f1)" = "$BIN_SHA" ] || { echo "binary sha256 differs from the registered one"; exit 1; }
[ "$(sha256sum "$ITEMS" | cut -d' ' -f1)" = "$ITEMS_SHA" ] || { echo "items sha256 differs from the registered one"; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT already exists: move it aside, a re-run must not mix rows"; exit 1; }
echo "$(ts) == D6a q4k start; host $(uname -srm); bin $(basename "$BIN"); load $(cut -d' ' -f1-3 /proc/loadavg); $(wc -l < "$ITEMS") items"
"$BIN" decide --model "$M9" --backend cpu --quant q4k --template bare-v1 --ctx 4096 --accept-slow -o "$OUT" "$ITEMS" 2> "$B/b0cmp-goinfer-bare-q4k.stderr"
echo "$(ts) == decide exit $?; rows: $(wc -l < "$OUT"); stderr tail:"
tail -3 "$B/b0cmp-goinfer-bare-q4k.stderr" | cut -c1-200
# An unattended run must not report success without its output: all 150 rows, every one by label scoring on the CPU path.
[ "$(wc -l < "$OUT")" -eq 150 ] || { echo "only $(wc -l < "$OUT") of 150 rows"; exit 1; }
if grep -q -i 'resident decode\|cuda-resident' "$B/b0cmp-goinfer-bare-q4k.stderr"; then echo "the run used a resident GPU path: not the CPU q4k measurement"; exit 1; fi
echo "$(ts) == DONE; analyse: python3 docs/measurements/decisions-d6a-2026-09-28/b0dist.py $OUT   (beside the int4 file: results/b0cmp-goinfer-bare.jsonl)"
