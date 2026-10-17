#!/bin/bash
# OP2b on CUDA, gate D (docs/tasks/task-forward-architecture-2026-10.md §4.7, "OP2b on CUDA"): each plant is a temporary edit of mamba.cu or
# nongated_act.cu with that module's PTX rebuilt, and gate K must go red on it; then the source and the PTX are restored and checked against git.
# The kernels are prebuilt PTX, so a plant cannot be a run-time switch (as OP1's G2-D was). Run it under the timing lock (the tests use the card):
#   python3 scripts/timing_lock.py run --label op2b-plants -- bash docs/measurements/op2b-cuda-2026-10-16/run-mamba-plants.sh
# The group-norm plant (one group over the full width where groups are asked for) is a launch plant inside TestMambaGNorm_groups itself.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)/cuda" || exit 2
TAGS='cuda goinfer_testhooks'
rc=0
plant() { # plant <name> <module> <old> <new> <test regexp>
  local name=$1 mod=$2 old=$3 new=$4 re=$5
  python3 - "$mod.cu" "$old" "$new" <<'PY' || { echo "PLANT NOT APPLIED: $name"; rc=1; return; }
import sys
f, old, new = sys.argv[1:4]
s = open(f).read()
assert s.count(old) == 1, (f, old, s.count(old))
open(f, "w").write(s.replace(old, new))
PY
  bash build_ptx.sh "$mod" > /dev/null 2>&1 || { echo "PLANT BUILD FAILED: $name"; rc=1; git checkout -q -- "$mod.cu" "testdata/$mod.ptx"; return; }
  out=$(go test -count=1 -tags "$TAGS" -run "$re" -v . 2>&1)
  if grep -q -- "--- FAIL" <<<"$out"; then echo "PLANT $name: RED, as it must be ($(grep -c -- '--- FAIL' <<<"$out") failing test(s); $(grep -E 'worst cosine|codes differ|scale' <<<"$out" | head -1 | sed 's/^ *//' | cut -c1-150))"
  else echo "PLANT $name: the gate STAYED GREEN"; rc=1; fi
  git checkout -q -- "$mod.cu" "testdata/$mod.ptx"
}
plant "conv taps shifted by one" mamba 's = __fadd_rn(s, __fmul_rn(convW[c * K + j], wv));' 's = __fadd_rn(s, __fmul_rn(convW[c * K + j + 1], wv));' '^TestMambaKernels_cpuParity$'
plant "softplus on dt dropped" mamba 'float dth = mb_softplus(__fadd_rn(dtRaw[dtBase + head], dtb));' 'float dth = __fadd_rn(dtRaw[dtBase + head], dtb);' '^TestMambaKernels_cpuParity$'
plant "squared ReLU as plain ReLU" nongated_act 'float d = __fmul_rn(x, x);' 'float d = x;' '^TestRelu2Quant_exact$'
# restored: the sources and PTX equal git's, and K is green again
git diff --quiet -- mamba.cu nongated_act.cu testdata/mamba.ptx testdata/nongated_act.ptx && echo "restored: sources and PTX equal git" || { echo "NOT RESTORED"; rc=1; }
go test -count=1 -tags "$TAGS" -run '^(TestMambaKernels_cpuParity|TestMambaGNorm_groups|TestRelu2Quant_exact)$' . > /dev/null 2>&1 && echo "K green again" || { echo "K NOT green after restore"; rc=1; }
exit $rc
