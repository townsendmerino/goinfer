#!/bin/bash
# B35-TL on nobara (README.md here): builds serve at the pinned commit with -tags cuda, runs gate.py, grades it. Queued on night.py (the job
# holds the timing lock) as `bash run-b35-toolloop.sh <sha>`, the commit in full. The runner's checkout may predate this script, so the
# queued command reads it from <sha> first (see README.md, "How it runs").
# Exit 0 PASS, 1 FAIL, 3 PARKED, 4 Q1-ONLY, 2 when it could not run (VOID).
set -u
SHA=${1:?the pinned commit}
B=$HOME/goinfer-bench/b35-toolloop; R=${SRC:-$HOME/goinfer-runner}; mkdir -p "$B/bin"
OUT=$HOME/goinfer-logs/b35-toolloop-2026-10/results-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
MODEL=$HOME/models/Qwen3.5-9B-Q4_K_M.gguf
[ -e "$MODEL" ] || { echo "VOID: $MODEL is missing"; exit 2; }
read -r load _ < /proc/loadavg
echo "=== $(date '+%F %T %Z') start: $SHA; $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader); load $load; MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB"
awk -v l="$load" 'BEGIN{exit !(l > 8.0)}' && { echo "VOID: started at load $load (> 8.0)"; exit 2; }
pgrep -x kwin_wayland >/dev/null && echo "desktop: kwin_wayland running (holds a CUDA context, about 16 MiB)"
git -C "$R" fetch -q origin
wt=$B/wt bin=$B/bin/serve-cuda-${SHA:0:8}
[ -d "$wt" ] && git -C "$R" worktree remove --force "$wt"
git -C "$R" worktree add -q --detach "$wt" "$SHA" || { echo "VOID: cannot check out $SHA"; exit 2; }
if [ ! -x "$bin" ]; then
	printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./cuda\n)\n' > "$wt/go.work"
	(cd "$wt/cuda" && GOWORK="$wt/go.work" CGO_ENABLED=0 go build -tags cuda -o "$bin" ./cmd/serve) || { echo "VOID: build failed"; exit 2; }
fi
HERE=$wt/docs/measurements/b35-toolloop-2026-10
python3 -u "$HERE/gate.py" --serve "$bin" --model "$MODEL" --out "$OUT" || { echo "VOID: gate.py exited $?"; exit 2; }
python3 "$HERE/grade.py" "$OUT/gate.json" | tee "$OUT/grade.txt"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T %Z') end, rc $rc; record $OUT"
git -C "$R" worktree remove --force "$wt" 2>/dev/null
exit "$rc"
