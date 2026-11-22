#!/usr/bin/env bash
# Evidence reruns at 36dbfbaf1: every quick -full test cell whose log shows the per-cell budget ran out ("panic: test timed out"),
# rerun with the same go test command (the cell log's first line) and -timeout 120m, in the same worktree, under the exclusive lock.
set -uo pipefail
D=$HOME/goinfer-logs/relgate-2026-11-21/quick-full
WT=$HOME/goinfer-bench/relgate-qf-wt
LOCK=$HOME/goinfer-runner/scripts/timing_lock.py
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
while pgrep -f "gate quick -full" >/dev/null; do sleep 30; done
R=$(ls -dt ${TMPDIR:-/tmp}/gate-quick-*/ | head -1)
cd "$WT" || exit 2
[ "$(git rev-parse HEAD)" = 36dbfbaf149aabdff43dce5766b49a65183113a7 ] || { echo "FATAL: not 36dbfbaf1"; exit 2; }
for L in "$R"test_*.log; do
  grep -q "panic: test timed out" "$L" || continue
  n=$(basename "$L" .log); cp "$L" "$D/$n-budget.log"
  cmd=$(python3 "$(dirname "$0")/cellcmd.py" "$L" 120m) || { echo "SKIP $n: cannot parse"; continue; }
  case "$cmd" in "go test -json -count=1 "*"-timeout 120m "*) ;; *) echo "SKIP $n: unexpected command: ${cmd:0:120}"; continue;; esac
  dir=$(sed -n 2p "$L" | sed -n "s/^# (in \(.*\))$/\1/p"); cd "${dir:-$WT}" || continue
  echo "== $n start $(date '+%F %T')"
  python3 "$LOCK" run --wait 1800 --label "relgate $n 120m (coordinator)" -- bash -c "$cmd" > "$D/$n-120m.json" 2>&1
  echo "rc=$? end $(date '+%F %T')"
  python3 -c '
import json,sys
c={"pass":0,"fail":0,"skip":0}; f=[]
for l in open(sys.argv[1]):
    try: e=json.loads(l)
    except Exception: continue
    a=e.get("Action")
    if e.get("Test") and a in c:
        c[a]+=1
        if a=="fail": f.append(e["Test"])
print(c, "failed:", f[:20])' "$D/$n-120m.json"
done
# One rerun of the environment red TestOtherGPUWindow_seesAnotherProcess (other GPU clients at 0.26 of wall time in run 1).
L="$R"test_backend_webgpu.log
if grep -q -- "--- FAIL: TestOtherGPUWindow_seesAnotherProcess" "$L"; then
  cp "$L" "$D/test_backend_webgpu-run1.log"
  dir=$(sed -n 2p "$L" | sed -n "s/^# (in \(.*\))$/\1/p"); cd "$dir" || exit 2
  cmd=$(python3 "$(dirname "$0")/cellcmd.py" "$L" 10m '^TestOtherGPUWindow_seesAnotherProcess$')
  echo "== webgpu one-test rerun: $cmd"
  python3 "$LOCK" run --wait 1800 --label "relgate webgpu one-test rerun (coordinator)" -- bash -c "${cmd/-json/-v}" > "$D/webgpu-othergpu-rerun.log" 2>&1
  echo "rc=$?"; grep -E "^(--- |ok|FAIL)" "$D/webgpu-othergpu-rerun.log"
fi
echo "ALL DONE $(date '+%F %T')"
