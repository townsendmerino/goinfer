#!/bin/bash
# OP1 on CUDA: TestQwen36_35B_cache alone, in a fresh process, with the v17 sidecar already built (owner, 2026-10-16), to settle the device gate's
# one void part. Builds the test binary at REV in a detached worktree (go test -c, so the process is the test and nothing wraps it), runs it, and
# logs the process's VmHWM and VmSwap and the per-process GPU memory every second until it exits.
#   python3 scripts/night.py add op1-35b-alone-cuda --est 15 ... -- env REV=<commit> bash docs/measurements/op1-cuda-2026-10/run-35b-alone.sh   (night.py holds the timing lock)
#   by hand: python3 scripts/timing_lock.py run --label op1-35b-alone -- env REV=c3a81a9f5026e0ccf1ef13d3e3f082049c440a53 bash docs/measurements/op1-cuda-2026-10/run-35b-alone.sh
# The card must have no other compute process; this refuses to start if one is there. The report line is "VmHWM ... against the 44,000 MB budget"
# (docs/measurements/b23-cuda-heavy-split-2026-10-15.md: Qwen3.6-35B-A3B, TestPagerDeterminism's reading 41,805 MB plus about 5%).
set -uo pipefail
SRC=${SRC:-$HOME/mycode/goinfer}
REV=${REV:?REV is required}
B=${B:-$HOME/goinfer-bench/op1-cuda-2026-10}
OUT=${OUT:-$B/35b-alone-$(date +%H%M)}
WT=$B/wt-35b-alone
mkdir -p "$OUT" || exit 2
others=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | grep -v kwin_wayland || true)
[ -z "$others" ] || { echo "REFUSED: another process holds the card:"; echo "$others"; exit 2; }
python3 - "$HOME/models/qwen3.6-35b-a3b-int4.giw" <<'PY' || exit 2
import struct, sys
b = open(sys.argv[1], "rb").read(0x49)
v = struct.unpack("<I", b[0x45:0x49])[0] if b[0x40:0x45] == b"GINFW" else -1
sys.exit(0 if v == 17 else (print("the 35B sidecar is v%d, not v17" % v) or 1))
PY
cd "$SRC" && git fetch -q origin
rev=$(git rev-parse --verify "$REV^{commit}") || exit 2
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$rev" || exit 2
cd "$WT" && go work init . ./gpu ./cuda ./metal ./demo/agent || exit 2
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the card check, the sidecar check and the worktree ($WT at $rev)"; cd "$SRC" && git worktree remove --force "$WT"; exit 0; }
(cd cuda && CGO_ENABLED=0 go test -c -tags 'cuda goinfer_testhooks' -o "$OUT/cuda.test" .) || { echo "FATAL: the test binary did not build"; exit 2; }
{ echo "rev: $rev"; echo "started: $(date '+%F %T %Z')"; echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"
  echo "swap before: $(free -m | awk '/Swap/{print $3" MB used of "$2}')"; echo "sidecar: $(ls -la "$HOME/models/qwen3.6-35b-a3b-int4.giw" | awk '{print $5, $6, $7, $8}')"; } | tee "$OUT/provenance.txt"
# exec: the backgrounded process IS the test binary, so /proc/$pid is the test's own VmHWM and VmSwap
( cd "$WT/cuda" && exec env GOINFER_HEAVY_TESTS=1 "$OUT/cuda.test" -test.run '^TestQwen36_35B_cache$' -test.v -test.timeout 30m ) > "$OUT/test.log" 2>&1 &
pid=$!
peak=0; swp=0
while kill -0 $pid 2>/dev/null; do
  read -r hwm rss sw < <(awk '/VmHWM/{h=$2}/VmRSS/{r=$2}/VmSwap/{s=$2}END{print h,r,s}' /proc/$pid/status 2>/dev/null)
  [ -n "${hwm:-}" ] && { [ "$hwm" -gt "$peak" ] && peak=$hwm; [ "${sw:-0}" -gt "$swp" ] && swp=$sw; }
  echo "$(date +%T) VmHWM=${hwm:-?} VmRSS=${rss:-?} VmSwap=${sw:-?} | $(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | tr '\n' ';')" >> "$OUT/sample.log"
  sleep 1
done
wait $pid; rc=$?
{ echo "test exit: $rc"; grep -E "^(--- |ok|FAIL|PASS)" "$OUT/test.log"
  echo "VmHWM peak: $((peak/1024)) MB   VmSwap peak: $((swp/1024)) MB   against the 44,000 MB budget: $([ $((peak/1024)) -le 44000 ] && echo UNDER || echo OVER)"
  echo "finished: $(date '+%F %T %Z')"; } | tee -a "$OUT/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
