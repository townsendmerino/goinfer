#!/bin/bash
# KV8a gate 5 with gate 4(b) (README.md here): KV0-H's opencode and Codex configuration rerun with the KV8a serve build, KV0-H's own harness
# pinned at bf023d40 (the commit KV0-H's study ran), then grade5.py against KV0-H's readings. It does not wait on gate 2(b) (the owner, 2026-10-20: gate 5 does not
# depend on 2(b)'s amendment); the record names 2(b)'s state when it ran. Queued on night.py (the job holds the timing lock) as
# `bash run-kv8a-gate5.sh <sha>`. Exit 0 PASS, 1 FAIL, 2 VOID or could not run.
set -u
SHA=${1:?the KV8a commit}
B=$HOME/goinfer-bench/kv8a; R=$HOME/mycode/goinfer; H=$B/wt-kv0h; W=$B/wt
OUT=$B/gate5-$(date +%F-%H%M); BODIES=$B/bodies-gate5-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start, pin $SHA; $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader); $(uptime)"
G2=$(ls -td "$B"/gate2b-*/ 2>/dev/null | head -1)
echo "gate 2(b) when this ran: $( [ -n "$G2" ] && grep -h '^VERDICT' "$G2/run.log" || echo 'no record') (${G2:-none})"
[ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" = 595.91.07 ] || { echo "VOID: driver is not KV0-H's 595.91.07"; exit 2; }
KPIN=bf023d40 # the driver commit KV0-H's study recorded (its START line, 2026-10-20 12:23:27)
[ -d "$H" ] || git -C "$R" worktree add -q --detach "$H" "$KPIN" || { echo "VOID: harness worktree"; exit 2; }
git -C "$H" checkout -q --force --detach "$KPIN" && [ "$(git -C "$H" rev-parse --short=8 HEAD)" = "$KPIN" ] || { echo "VOID: the harness worktree is not at $KPIN"; exit 2; }
git -C "$R" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$R" fetch -q "$B/kv8a.bundle" "+kv8a-2026-10:refs/mac/kv8a-2026-10" || { echo "VOID: bundle"; exit 2; }
git -C "$W" checkout -q --force --detach "$SHA" || { echo "VOID: worktree"; exit 2; }
BIN=$B/serve-cuda-${SHA:0:8}
[ -x "$BIN" ] || (printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./cuda\n)\n' > "$W/go.work" && cd "$W/cuda" && GOWORK="$W/go.work" CGO_ENABLED=0 go build -tags cuda -o "$BIN" ./cmd/serve) || { echo "VOID: serve build"; exit 2; }
K=$H/docs/measurements/kv0h-2026-10
[ -f "$K/plan-study-h20.json" ] || { echo "VOID: no plan-study-h20.json at $KPIN"; exit 2; }
plan() { # $1 out plan; $2 sessions json (empty: the study's)
	python3 - "$K/plan-study-h20.json" "$1" "$BIN" "$2" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
p["serve"] = sys.argv[3]
if sys.argv[4]:
    p["configs"][0]["sessions"] = json.loads(sys.argv[4])
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
}
CFG=h-qwen35-9b-20k
collect() { # $1 plan; $2 out; $3 bodies. run-kv0h.sh exits 0 whatever its sessions did, so its END line and the stats are checked here.
	bash "$K/run-kv0h.sh" "$1" "$2" "$3" | tee "$2/kv0h.log"
	grep -q '^END .* sessions rc=0$' "$2/kv0h.log" && [ -s "$2/stats-h-$CFG.json" ] || { echo "VOID: the KV0-H collection did not complete ($2)"; exit 2; }
}
plan "$OUT/plan.json" "" || { echo "VOID: plan"; exit 2; }
collect "$OUT/plan.json" "$OUT" "$BODIES"
G=(python3 "$W/docs/measurements/kv8a-2026-10/grade5.py" --base "$W/docs/measurements/kv0h-2026-10/study-h20-stats-h.json")
"${G[@]}" --stats "$OUT/stats-h-$CFG.json" --serve-log "$OUT/logs/serve-$CFG.log" | tee "$OUT/grade5.txt"
rc=${PIPESTATUS[0]}
if [ "$rc" = 3 ]; then # the stopping rule: one chain session per harness, once
	O2=$OUT/extend; mkdir -p "$O2"
	plan "$O2/plan.json" '[{"agent":"opencode","task":"chain","rep":3},{"agent":"codex","task":"chain","rep":3}]' || { echo "VOID: plan"; exit 2; }
	collect "$O2/plan.json" "$O2" "$BODIES-extend"
	"${G[@]}" --stats "$OUT/stats-h-$CFG.json" --stats "$O2/stats-h-$CFG.json" --serve-log "$OUT/logs/serve-$CFG.log" --serve-log "$O2/logs/serve-$CFG.log" | tee "$OUT/grade5.txt"
	rc=${PIPESTATUS[0]}
	[ "$rc" = 3 ] && rc=2
fi
echo "=== $(date '+%F %T %Z') end, rc $rc; record $OUT"
exit "$rc"
