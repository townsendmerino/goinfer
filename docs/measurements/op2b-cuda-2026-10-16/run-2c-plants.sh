#!/bin/bash
# OP2b on CUDA, step 2c's planted defects (docs/tasks/task-forward-architecture-2026-10.md §4.7): each is a temporary source edit, the named test must go red,
# then the file is restored from git. Run under the timing lock (the tests use the card):
#   python3 scripts/timing_lock.py run --label op2b-2c-plants -- bash docs/measurements/op2b-cuda-2026-10-16/run-2c-plants.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)/cuda" || exit 2
TAGS='cuda goinfer_testhooks'
rc=0
# each plant restores its file with `git checkout`, which would discard uncommitted work in it: commit first
git diff --quiet HEAD -- . || { echo "REFUSED: cuda/ has uncommitted changes; commit them first (a plant restores its file from git)"; exit 2; }
plant() { # plant <name> <file> <old> <new> <test regexp>
  local name=$1 f=$2 old=$3 new=$4 re=$5
  python3 - "$f" "$old" "$new" <<'PY' || { echo "PLANT NOT APPLIED: $name"; rc=1; return; }
import sys
f, old, new = sys.argv[1:4]
s = open(f).read()
assert s.count(old) == 1, (f, old, s.count(old))
open(f, "w").write(s.replace(old, new))
PY
  out=$(go test -count=1 -tags "$TAGS" -run "$re" -v . 2>&1)
  if grep -q -- "--- FAIL" <<<"$out"; then echo "PLANT $name: RED, as it must be ($(grep -c -- '--- FAIL' <<<"$out") failing, e.g. $(grep -m1 -E '_test.go:[0-9]+:' <<<"$out" | sed 's/^ *//' | cut -c1-150))"
  else echo "PLANT $name: the gate STAYED GREEN"; rc=1; fi
  git checkout -q -- "$f"
}
G='^TestNemotronResidentCUDA_guards$'
P='^TestNemotronResidentCUDA_pricing$'
plant "the batched step's recurrent check left DeltaNet-only" batchstep.go 'r.recurrent() || r.KVSlots() < 2' 'r.dnet != nil || r.KVSlots() < 2' "$G"
plant "CopyKV's recurrent check left DeltaNet-only" kvcopy.go 'case r.recurrent():' 'case r.dnet != nil:' "$G"
plant "the prefix snapshot's recurrent check left DeltaNet-only" kvsnap.go 'case r.recurrent():' 'case r.dnet != nil:' "$G"
plant "the verify pass left DeltaNet-only" mamba_resident.go 'return r.prefillReady && !r.recurrent()' 'return r.prefillReady && r.dnet == nil' "$G"
plant "the batched prefill no longer names the Mamba-2 mixer" prefill.go 'case decoder.MixerMamba:
		return "Mamba-2 mixer"' 'case decoder.MixerMamba:' "$G"
plant "a Mamba or feed-forward block keeps its attention geometry (only DeltaNet zeroed)" backend.go 'if h.isDeltaNet || h.mixer == decoder.MixerMamba || h.mixer == decoder.MixerNone {' 'if h.isDeltaNet {' "$P"
plant "the plan prices no Mamba state" mamba_resident.go 'if p == nil || layers <= 0 {' 'if true || p == nil || layers <= 0 {' "$P"
git diff --quiet -- . && echo "restored: sources equal git" || { echo "NOT RESTORED"; rc=1; }
exit $rc
