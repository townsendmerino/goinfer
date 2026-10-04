#!/usr/bin/env bash
# Night job: int4mix gate M4 (docs/tasks/task-metal-int8-2026-10.md, "Slice 4: int4mix"; adapted from run-m4.sh), pre-registered there on
# 2026-10-04 before any graded run: the §3.2 pooled fidelity gate (TestPrefillGateVsReference) on the 1.5B, prompt set
# A, decision set K = 256, 512, 1024, confirmation cells off, both arms on the native int4mix path with precise math,
# scored against the CPU f32 references in ~/goinfer-logs/prefill-ref (S-K*). A pool that does not ship exits non-zero:
# that is the verdict, recorded.
#
# Runs a test binary built at REV from a clean tree, from a worktree pinned at REV (the gate reads ../testdata):
#   go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-int8-2026-10/metal-<REV>.test ./metal/
#   git worktree add --detach ~/goinfer-bench/metal-int8-2026-10/wt-<REV> <REV>
# Queued with:
#   python3 scripts/night.py add metal-int8-m4 --est 20 --by "Claude, Metal int4mix M4" \
#     --doc docs/tasks/task-metal-int8-2026-10.md -- bash docs/measurements/metal-int8-2026-10/run-m4.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-int8-2026-10/m4/.
set -uo pipefail
REV=e647ac86
BASE=$HOME/goinfer-bench/metal-int8-2026-10
BIN=$BASE/metal-$REV.test
WT=$BASE/wt-$REV
LOG=$HOME/goinfer-logs/metal-int8-2026-10/m4
REF=$HOME/goinfer-logs/prefill-ref
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
[ -d "$WT/testdata/prefill-gate-prose-a" ] || { echo "missing worktree $WT"; exit 1; }
for k in 256 512 1024; do ls "$REF"/S-K$k-p*.bin >/dev/null 2>&1 || { echo "missing references $REF/S-K$k-*"; exit 1; }; done
cd "$WT/metal"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"
env GOINFER_HEAVY_TESTS=1 GOINFER_METAL_GATE_QUANT=int4mix GOINFER_METAL_GATE_MODELS=S GOINFER_METAL_GATE_CONFIRM=0 \
  "$BIN" -test.v -test.count=1 -test.timeout 60m -test.run '^TestPrefillGateVsReference$' 2>&1 | tee "$LOG/m4-1.5b.log"
s=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), exit $s" | tee -a "$LOG/provenance.txt"
exit "$s"
