#!/bin/bash
# OP2b on CUDA, step 2a's planted defects (docs/tasks/task-forward-architecture-2026-10.md §4.7): each is a temporary source edit, the named test must go red,
# then the file is restored from git. Run under the timing lock (the tests use the card):
#   python3 scripts/timing_lock.py run --label op2b-2a-plants -- bash docs/measurements/op2b-cuda-2026-10-16/run-2a-plants.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)/cuda" || exit 2
TAGS='cuda goinfer_testhooks'
rc=0
plant() { # plant <name> <file> <old> <new> <env> <test regexp>
  local name=$1 f=$2 old=$3 new=$4 envs=$5 re=$6
  python3 - "$f" "$old" "$new" <<'PY' || { echo "PLANT NOT APPLIED: $name"; rc=1; return; }
import sys
f, old, new = sys.argv[1:4]
s = open(f).read()
assert s.count(old) == 1, (f, old, s.count(old))
open(f, "w").write(s.replace(old, new))
PY
  out=$(env $envs go test -count=1 -tags "$TAGS" -run "$re" -v . 2>&1)
  if grep -q -- "--- FAIL" <<<"$out"; then echo "PLANT $name: RED, as it must be ($(grep -c -- '--- FAIL' <<<"$out") failing, e.g. $(grep -m1 -E '_test.go:[0-9]+:' <<<"$out" | sed 's/^ *//' | cut -c1-130))"
  else echo "PLANT $name: the gate STAYED GREEN"; rc=1; fi
  git checkout -q -- "$f"
}
plant "cudaLayer.mixer filled as attention for every layer" backend.go 'L := cudaLayer{idx: l, mixer: h.mixer, ffn: h.ffn}' 'L := cudaLayer{idx: l, mixer: decoder.MixerAttention, ffn: h.ffn}' "" '^TestLayerOps_agreeWithTheFamilyPredicates$'
plant "launchToken dispatches no layer to the DeltaNet mixer" resident.go 'if Ly.mixer == decoder.MixerDeltaNet {
			// Gated-DeltaNet mixer: replaces segA' 'if false {
			// Gated-DeltaNet mixer: replaces segA' "GOINFER_DNET_PARITY_CUDA=1" '^TestQwen35ResidentParityCUDA$'
git diff --quiet -- backend.go resident.go && echo "restored: sources equal git" || { echo "NOT RESTORED"; rc=1; }
exit $rc
