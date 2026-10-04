#!/usr/bin/env bash
# Night job: F3′ on the 1.5B (docs/tasks/task-metal-int8-2026-10.md, "F3′" and "F3′ amendment"), pre-registered before
# the run: TestW8Native_F3amended_closerToF32 on the 1.5B coder, every arm on the Mac, the CPU f32 reference read from
# the file nobara wrote (TestW8F3Reference_write, amd64, sha256 ad1df1ef0c383c67...). Bar: pooled mean KL(f32 || Metal
# int8int8) <= 1.10 x KL(f32 || CPU int8int8, f16 KV).
#
# Runs a test binary built at REV from a clean tree, from a worktree pinned at REV (the test reads ../testdata):
#   go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-int8-2026-10/metal-<REV>.test ./metal/
#   git worktree add --detach ~/goinfer-bench/metal-int8-2026-10/wt-<REV> <REV>
# Queued with:
#   python3 scripts/night.py add metal-int8-f3-1.5b --est 10 --by "Claude, Metal int8 F3′ 1.5B" \
#     --doc docs/tasks/task-metal-int8-2026-10.md -- bash docs/measurements/metal-int8-2026-10/run-f3amended-1.5b.sh
# Logs: ~/goinfer-logs/metal-int8-2026-10/f3-1.5b/.
set -uo pipefail
REV=518c360c
BASE=$HOME/goinfer-bench/metal-int8-2026-10
BIN=$BASE/metal-$REV.test
WT=$BASE/wt-$REV
REF=$HOME/goinfer-logs/metal-int8-2026-10/f3ref-1.5b-amd64.gob.gz
LOG=$HOME/goinfer-logs/metal-int8-2026-10/f3-1.5b
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
[ -d "$WT/testdata/prefill-gate-prose-a" ] || { echo "missing worktree $WT"; exit 1; }
[ "$(shasum -a 256 "$REF" | cut -c1-16)" = "ad1df1ef0c383c67" ] || { echo "reference $REF missing or not the one nobara wrote"; exit 1; }
cd "$WT/metal"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ref:      $REF (sha256 $(shasum -a 256 "$REF" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"
env GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_MODEL=$M15 GOINFER_W8_F3_REF_IN=$REF "$BIN" -test.v -test.count=1 -test.timeout 40m \
  -test.run '^TestW8Native_F3amended_closerToF32$' 2>&1 | tee "$LOG/f3amended-1.5b.log"
s=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), exit $s" | tee -a "$LOG/provenance.txt"
exit "$s"
