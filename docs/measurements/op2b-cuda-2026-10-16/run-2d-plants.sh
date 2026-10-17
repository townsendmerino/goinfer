#!/bin/bash
# OP2b on CUDA, step 2d's planted defects (docs/tasks/task-forward-architecture-2026-10.md §4.7): each is a temporary source edit, the named test must go red,
# then the file is restored from git. Run under the timing lock (the tests use the card), on a COMMITTED tree (a plant restores its file with git checkout):
#   python3 scripts/timing_lock.py run --label op2b-2d-plants -- bash docs/measurements/op2b-cuda-2026-10-16/run-2d-plants.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)/cuda" || exit 2
git diff --quiet HEAD -- . || { echo "REFUSED: cuda/ has uncommitted changes; commit them first (a plant restores its file from git)"; exit 2; }
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
  if grep -q -- "--- FAIL" <<<"$out"; then echo "PLANT $name: RED, as it must be ($(grep -c -- '--- FAIL' <<<"$out") failing, e.g. $(grep -m1 -E '_test.go:[0-9]+:' <<<"$out" | sed 's/^ *//' | cut -c1-140))"
  else echo "PLANT $name: the gate STAYED GREEN"; rc=1; fi
  git checkout -q -- "$f"
}
L='^TestLayerOps_buildTheKindsTheFamilyHas$'
plant "cudaLayer.mixer filled as attention for every layer" backend.go 'L := cudaLayer{idx: l, mixer: h.mixer, ffn: h.ffn}' 'L := cudaLayer{idx: l, mixer: decoder.MixerAttention, ffn: h.ffn}' "" "$L"
plant "launchToken dispatches no layer to the DeltaNet mixer" resident.go 'if Ly.mixer == decoder.MixerDeltaNet {
			// Gated-DeltaNet mixer: replaces segA' 'if false {
			// Gated-DeltaNet mixer: replaces segA' "GOINFER_DNET_PARITY_CUDA=1" '^TestQwen35ResidentParityCUDA$'
plant "isDeltaNet() reads false" mamba_resident.go 'func (L *cudaLayer) isDeltaNet() bool { return L.mixer == decoder.MixerDeltaNet }' 'func (L *cudaLayer) isDeltaNet() bool { return false }' "GOINFER_DNET_PARITY_CUDA=1" '^(TestQwen35ResidentParityCUDA|TestLayerOps_buildTheKindsTheFamilyHas)$'
plant "isMLA() reads false" mamba_resident.go 'func (L *cudaLayer) isMLA() bool      { return L.mixer == decoder.MixerMLA }' 'func (L *cudaLayer) isMLA() bool      { return false }' "" '^(TestMLAResidentParityCUDA)$'
git diff --quiet -- . && echo "restored: sources equal git" || { echo "NOT RESTORED"; rc=1; }
exit $rc
