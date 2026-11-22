#!/usr/bin/env bash
# KV10 on the frozen text (README.md, "The frozen text"): depths 3900 and 65536 of TestKV10_longContextFidelity, one test binary
# built from a clean worktree, run under the box's exclusive timing lock. Usage: run-frozen.sh <clean worktree> <log dir>
set -uo pipefail
W=$1 D=$2
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH CGO_ENABLED=0
mkdir -p "$D"
cd "$W" || exit 2
[ -z "$(git status --porcelain)" ] || { echo "VOID: $W is not clean"; git status --porcelain; exit 2; }
echo "rev $(git rev-parse HEAD)"
go test -c -tags 'cuda goinfer_testhooks' -o "$D/kv10.test" ./backend/cuda || { echo "FATAL: build"; exit 2; }
exec python3 scripts/timing_lock.py run --label "KV10 frozen text" -- bash -c '
  D=$1
  echo "start $(date "+%F %T %Z"); load $(cut -d" " -f1-3 /proc/loadavg); $(grep MemAvailable /proc/meminfo)"
  nvidia-smi --query-gpu=driver_version,memory.used,memory.total --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  cd backend/cuda || exit 2
  GOINFER_HEAVY_TESTS=1 GOINFER_KV10_DEPTHS=3900,65536 KV10_OUT="$D/fidelity-frozen.jsonl" \
    "$D/kv10.test" -test.run "^TestKV10_longContextFidelity$" -test.v -test.timeout 45m 2>&1 | tee "$D/test.log"
  rc=${PIPESTATUS[0]}
  echo "end $(date "+%F %T %Z"); exit $rc"
  exit $rc
' _ "$D"
