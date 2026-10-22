#!/bin/bash
# KV8a's 28K Claude Code chain on CUDA (README.md here, "The 28K Claude Code chain on CUDA, on today's main"): builds serve at the pinned
# main commit, takes gate2b.py from the Metal branch's pinned commit (it has --backend and --configs), checks the fit and the box, and runs
# the cc config only. Run under the timing lock as `bash run-kv8a-cc28k-cuda.sh`. Exit 0 PASS, 1 FAIL, 2 VOID or could not run.
set -u
SHA=799b98df0; HARNESS=279bc973f
R=$HOME/mycode/goinfer; K=$HOME/goinfer-bench/kv0h-2026-10; B=$HOME/goinfer-bench/kv8a-cc28k
MODEL=$HOME/models/Qwen3.5-9B-Q4_K_M.gguf
OUT=$B/run-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
export PATH=/usr/local/go/bin:$PATH
echo "=== $(date '+%F %T %Z') start, serve pin $SHA, harness $HARNESS; $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader); $(uptime)"
[ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" = 595.91.07 ] || { echo "VOID: driver is not 595.91.07"; exit 2; }
OTHERS=$(nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | grep -v kwin_wayland | grep -c .)
[ "$OTHERS" = 0 ] || { echo "VOID: another process holds the card"; exit 2; }
for f in "$MODEL" "$K/ids.key" "$K/pilot/ids.jsonl" "$K/bodies-pilot/pilot-qwen35-9b-28k-cc-claude-code-chain-r1"; do [ -e "$f" ] || { echo "VOID: $f is missing"; exit 2; }; done
W=$B/wt
[ -d "$W" ] || git -C "$R" worktree add -q --detach "$W" "$SHA" || { echo "VOID: worktree"; exit 2; }
git -C "$W" checkout -q --force --detach "$SHA" || { echo "VOID: checkout"; exit 2; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./cuda\n)\n' > "$W/go.work"
BIN=$B/serve-cuda-${SHA:0:8}
[ -x "$BIN" ] || (cd "$W/cuda" && GOWORK="$W/go.work" CGO_ENABLED=0 go build -tags cuda -o "$BIN" ./cmd/serve) || { echo "VOID: serve build"; exit 2; }
(cd "$W" && GOWORK="$W/go.work" go test -count=1 -v -run '^TestLogitBias' ./internal/serveapp/ 2>&1 | grep -E '^(--- |ok|FAIL)') | tee /dev/stderr | grep -q '^--- PASS: TestLogitBias_reachesTheSamplerOnEveryRoute' || { echo "VOID: logit_bias does not reach the sampler at this pin"; exit 2; }
git -C "$R" show "$HARNESS:docs/measurements/kv8a-2026-10/gate2b.py" > "$OUT/gate2b.py" || { echo "VOID: harness"; exit 2; }
echo "== fit check: the server's own plan at -ctx 32768"
"$BIN" --model "$MODEL" --backend cuda --ctx 32768 --addr 127.0.0.1:18299 --kv-sessions 1 > "$OUT/fit.log" 2>&1 &
FP=$!
for _ in $(seq 180); do grep -q 'goinfer serving on' "$OUT/fit.log" && break; kill -0 $FP 2>/dev/null || break; sleep 1; done
kill $FP 2>/dev/null; wait $FP 2>/dev/null
grep -E 'decode path|fit:|KV plan|refus|checkpoint' "$OUT/fit.log" | head -20
grep -q 'goinfer serving on' "$OUT/fit.log" && grep -q 'decode path: cuda-resident' "$OUT/fit.log" || { echo "VOID: the fit check did not load the 9B resident on CUDA at -ctx 32768"; exit 2; }
NEED_KB=$(( (7 + 2) * 1024 * 1024 ))
read -r L1 _ < /proc/loadavg; AVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
echo "== start check: load1 $L1, MemAvailable $((AVAIL / 1024)) MiB (need ${NEED_KB} kB: the load's 7 GB plus 2 GB)"
awk -v l="$L1" 'BEGIN{exit !(l > 8.0)}' && { echo "VOID: load average $L1 above 8.0"; exit 2; }
[ "$AVAIL" -ge "$NEED_KB" ] || { echo "VOID: MemAvailable below the need plus 2 GB"; exit 2; }
python3 -u "$OUT/gate2b.py" --serve "$BIN" --model "$MODEL" --key "$K/ids.key" --bodies "$K/bodies-pilot" --ids "$K/pilot/ids.jsonl" \
	--out "$OUT" --backend cuda --configs cc
echo "== gate2b.py exited $? (its own verdict floors at 15 tested pairs, which this chain cannot reach; the cell is graded below)"
python3 - "$OUT/gate2b.json" <<'PY'
import json, sys
try:
    pairs = json.load(open(sys.argv[1]))["pairs"]
except Exception as e:
    print(f"VERDICT VOID (no complete gate2b.json: {e})"); sys.exit(2)
if len(pairs) != 7 or any("same" not in p for p in pairs):
    print(f"VERDICT VOID ({len(pairs)} of 7 pairs graded)"); sys.exit(2)
tested = [p for p in pairs if p["tested"]]
bad = [p for p in pairs if not p["same"] or p["cold_reused"] != 0]
for p in pairs:
    print(f"pair {p['pair']}: turn 1 {p['t1_len']}, turn 2 {p['t2_len']}, warm reused {p['warm_reused']}, tested {p['tested']}, warm == cold {p['same']}")
if bad:
    print(f"VERDICT FAIL ({len(tested)} tested, {len(bad)} of 7 differ)"); sys.exit(1)
if not tested:
    print("VERDICT VOID (no pair tested the restore)"); sys.exit(2)
print(f"VERDICT PASS ({len(tested)} tested, 0 of 7 differ)")
PY
rc=$?
echo "=== $(date '+%F %T %Z') end, rc $rc; record $OUT"
exit $rc
