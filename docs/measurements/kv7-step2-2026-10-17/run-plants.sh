#!/bin/bash
# KV7 step 2's planted defects (docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md §8): each is a temporary source edit, the named test must go red, then
# the files are restored from git. Run on a COMMITTED tree (a plant restores with git checkout), under the timing lock for the CUDA ones:
#   python3 scripts/timing_lock.py run --label kv7s2-plants -- bash docs/measurements/kv7-step2-2026-10-17/run-plants.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2
git diff --quiet HEAD -- . || { echo "REFUSED: uncommitted changes; commit them first (a plant restores its files from git)"; exit 2; }
TAGS='cuda goinfer_testhooks'
rc=0
# plant <name> <test dir: decoder|cuda> <test regexp> <file> <old> <new> [<file> <old> <new> ...]
plant() {
  local name=$1 dir=$2 re=$3; shift 3
  local files=()
  while [ $# -ge 3 ]; do
    python3 - "$1" "$2" "$3" <<'PY' || { echo "PLANT NOT APPLIED: $name ($1)"; rc=1; git checkout -q -- . ; return; }
import sys
f, old, new = sys.argv[1:4]
s = open(f).read()
assert s.count(old) == 1, (f, old, s.count(old))
open(f, "w").write(s.replace(old, new))
PY
    files+=("$1"); shift 3
  done
  if [ "$dir" = cuda ]; then out=$(cd cuda && go test -count=1 -tags "$TAGS" -run "$re" -v . 2>&1)
  else out=$(go test -count=1 -tags goinfer_testhooks -run "$re" -v ./decoder/ 2>&1); fi
  if grep -q -- "--- FAIL" <<<"$out"; then echo "PLANT $name: RED, as it must be ($(grep -c -- '--- FAIL' <<<"$out") failing, e.g. $(grep -m1 -E '_test.go:[0-9]+:' <<<"$out" | sed 's/^ *//' | cut -c1-150))"
  else echo "PLANT $name: the gate STAYED GREEN"; rc=1; fi
  git checkout -q -- "${files[@]}"
}
D='^TestKVSnapshotBoundary_'
plant "1 the snapshot taken one pass late (CUDA, the ring moved on)" cuda '^TestKV7Mark_' cuda/prefill.go 'var markLatePlantForTest bool' 'var markLatePlantForTest = true'
plant "2 the boundary rounded UP past the pass boundary" decoder "$D" decoder/kvsnap.go 'q = sc.Boundary / align * align' 'q = (sc.Boundary + align - 1) / align * align'
plant "3 the boundary ignored: step one's end snapshot for a long-tailed prompt" decoder "$D" decoder/kvsnap.go 'if sn == nil || !hasMark || sc.Boundary <= 0' 'if true || sn == nil || !hasMark || sc.Boundary <= 0'
plant "4 restore considers only the longest match" decoder "$D" decoder/kvsnap.go '	return out
}

// covering' '	return out[:min(1, len(out))]
}

// covering'
plant "5 a second snapshot of the same prompt: the end snapshot also taken after the mark" decoder "$D" decoder/kvsnap.go 'if acq.markFired || acq.snapRows > 0 {' 'if false {' decoder/kvsnap.go '; q > 0 && x.h.Serves(q) {' '; false && q > 0 && x.h.Serves(q) {'
plant "6 the ring base not set on a restore (CUDA, through the mid-prefill path)" cuda '^TestKV7Mark_' cuda/kvsnap.go '		r.kvBases[slot] = lo
		if slot == r.kvSlot {' '		if slot == r.kvSlot {'
git diff --quiet -- . && echo "restored: sources equal git" || { echo "NOT RESTORED"; rc=1; }
exit $rc
