#!/bin/bash
# B35's end-to-end gate on nobara (README.md here): builds the B35 head and its base on main with -tags cuda, runs gate.py, grades it.
# Queued on night.py (the job holds the timing lock) as `bash run-b35-e2e.sh <B35 sha> <base sha>`, the two pinned commits in full.
# Exit 0 on PASS, 1 on FAIL, 3 on PARKED, 2 when it could not run.
set -u
B35=${1:?the B35 commit}
BASE=${2:?the base commit}
B=$HOME/goinfer-bench/b35-e2e; R=$HOME/mycode/goinfer; mkdir -p "$B/bin"
OUT=$B/results-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
MODEL=$HOME/models/Qwen3.5-9B-Q4_K_M.gguf
[ -e "$MODEL" ] || { echo "VOID: $MODEL is missing"; exit 2; }
echo "=== $(date '+%F %T %Z') start: B35 $B35, base $BASE; $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader); $(uptime)"
git -C "$R" fetch -q origin main
git -C "$R" cat-file -e "$B35^{commit}" 2>/dev/null || git -C "$R" fetch -q "$B/b35.bundle" "+b35-recurrent-checkpoint-2026-10:refs/mac/b35-recurrent-checkpoint-2026-10" || { echo "VOID: bundle"; exit 2; }
build() { # $1 name, $2 sha
	local wt=$B/wt-$1 bin=$B/bin/serve-cuda-$1-${2:0:8}
	[ -x "$bin" ] && return 0
	if [ -d "$wt" ]; then git -C "$wt" checkout -q --force --detach "$2"; else git -C "$R" worktree add -q --detach "$wt" "$2"; fi || return 1
	[ "$(git -C "$wt" rev-parse HEAD)" = "$2" ] || return 1
	printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./cuda\n)\n' > "$wt/go.work"
	(cd "$wt/cuda" && GOWORK="$wt/go.work" CGO_ENABLED=0 go build -tags cuda -o "$bin" ./cmd/serve)
}
build b35 "$B35" || { echo "VOID: the B35 build failed"; exit 2; }
build base "$BASE" || { echo "VOID: the base build failed"; exit 2; }
HERE=$B/wt-b35/docs/measurements/b35-e2e-2026-10-20
python3 -u "$HERE/gate.py" --b35 "$B/bin/serve-cuda-b35-${B35:0:8}" --base "$B/bin/serve-cuda-base-${BASE:0:8}" --model "$MODEL" --out "$OUT" || { echo "VOID: gate.py exited $?"; exit 2; }
python3 "$HERE/grade.py" "$OUT/gate.json" | tee "$OUT/grade.txt"
rc=${PIPESTATUS[0]}
echo "=== $(date '+%F %T %Z') end, rc $rc; record $OUT"
exit "$rc"
