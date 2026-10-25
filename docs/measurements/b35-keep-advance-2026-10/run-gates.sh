#!/bin/bash
# The B35 follow-up's nobara gates (README.md here): G1 (cuda/b35_snapshot_test.go at the change), G2 (B35-TL at the change), G3 (B35's own
# end-to-end gate, the change against main). Queued on night.py (the job holds the timing lock):
#   bash run-gates.sh <change sha> <main sha> <B35-TL scripts sha>
# B35-TL's gate.py and grade.py are read from the third commit, which holds them. Exit 0 when every gate passed, 1 otherwise.
set -u
FIX=${1:?the change commit}
MAIN=${2:?the main commit}
TL=${3:?the commit holding the B35-TL scripts}
R=${SRC:-$HOME/goinfer-runner}
B=$HOME/goinfer-bench/b35-keep-advance; mkdir -p "$B/bin"
OUT=$HOME/goinfer-logs/b35-keep-advance-2026-10/gates-$(date +%F-%H%M); mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
export PATH=/usr/local/go/bin:$PATH
read -r load _ < /proc/loadavg
echo "=== $(date '+%F %T %Z') start: change $FIX, main $MAIN, B35-TL scripts $TL; $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader); load $load; MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB"
awk -v l="$load" 'BEGIN{exit !(l > 8.0)}' && { echo "VOID: started at load $load (> 8.0)"; exit 2; }
pgrep -x kwin_wayland >/dev/null && echo "desktop: kwin_wayland running (holds a CUDA context, about 16 MiB)"
git -C "$R" fetch -q origin main b35-keep-advance-2026-10 kapitsa-steps-2026-10 || { echo "VOID: fetch failed"; exit 2; }
wt=$B/wt
[ -d "$wt" ] && git -C "$R" worktree remove --force "$wt"
git -C "$R" worktree add -q --detach "$wt" "$FIX" || { echo "VOID: cannot check out $FIX"; exit 2; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./cuda\n)\n' > "$wt/go.work"
rc=0

echo "== G1: cuda/b35_snapshot_test.go at the change ($(date '+%T'))"
(cd "$wt/cuda" && GOWORK="$wt/go.work" CGO_ENABLED=0 go test -tags 'cuda goinfer_testhooks' -count=1 -run 'TestB35' -v -timeout 10m . > "$OUT/g1.log" 2>&1)
grep -E '^(--- |ok|FAIL)' "$OUT/g1.log"
if grep -q -- '--- FAIL\|^FAIL' "$OUT/g1.log" || grep -q -- '--- SKIP' "$OUT/g1.log" || ! grep -q -- '--- PASS' "$OUT/g1.log"; then echo "G1 FAILED (a skip counts as a failure)"; rc=1; else echo "G1 PASS"; fi

echo "== G2: B35-TL at the change ($(date '+%T'))"
bin=$B/bin/serve-cuda-${FIX:0:8}
(cd "$wt/cuda" && GOWORK="$wt/go.work" CGO_ENABLED=0 go build -tags cuda -o "$bin" ./cmd/serve) || { echo "VOID: build failed"; exit 2; }
mkdir -p "$B/tl"
for f in gate.py grade.py; do git -C "$R" show "$TL:docs/measurements/b35-toolloop-2026-10/$f" > "$B/tl/$f" || { echo "VOID: no $f at $TL"; exit 2; }; done
python3 -u "$B/tl/gate.py" --serve "$bin" --model "$HOME/models/Qwen3.5-9B-Q4_K_M.gguf" --out "$OUT/g2" || { echo "VOID: gate.py exited $?"; exit 2; }
python3 "$B/tl/grade.py" "$OUT/g2/gate.json" | tee "$OUT/g2/grade.txt"
g2=${PIPESTATUS[0]}
[ "$g2" = 0 ] && echo "G2 PASS" || { echo "G2 FAILED (grade exit $g2)"; rc=1; }

echo "== G3: B35's end-to-end gate, the change against main ($(date '+%T'))"
mkdir -p "$HOME/goinfer-bench/b35-e2e"
git -C "$HOME/mycode/goinfer" fetch -q origin b35-keep-advance-2026-10 || { echo "VOID: the e2e checkout cannot fetch the change"; exit 2; }
cp "$wt/docs/measurements/b35-e2e-2026-10-20/run-b35-e2e.sh" "$B/run-b35-e2e.sh"
bash "$B/run-b35-e2e.sh" "$FIX" "$MAIN" > "$OUT/g3.log" 2>&1
g3=$?
grep -E 'VERDICT|^FAIL|^PARK|VOID' "$OUT/g3.log" | tail -8
[ "$g3" = 0 ] && echo "G3 PASS" || { echo "G3 FAILED (exit $g3)"; rc=1; }

git -C "$R" worktree remove --force "$wt" 2>/dev/null
echo "=== $(date '+%F %T %Z') end, rc $rc; record $OUT"
exit $rc
