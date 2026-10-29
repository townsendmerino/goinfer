#!/usr/bin/env bash
# SB12 (Gemini on #415 item 3): the seven tests sb12-memtests §2 left as found, each in its own process (-p 1), with the
# subprocess arms also run directly so their own end-of-run release check prints. Loaders hold the timing lock shared;
# TestKSA1_gemvTiming is a timed reading and holds it exclusively, as its comment directs.
set -u
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
T="go test -tags cuda,goinfer_testhooks -count=1 -p 1 -timeout 10m -v ./cuda/"
one() { # label, env..., -- , run regex
  local label=$1; shift; local envs=(); while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
  echo "== $label start $(date '+%H:%M:%S') load $(cut -d' ' -f1 /proc/loadavg) env ${envs[*]:-none}"
  env "${envs[@]}" python3 scripts/timing_lock.py run ${LOCK:---shared} --label "sb12-seven-$label" -- $T -run "$1" 2>&1 |
    grep -E '^(=== RUN|--- |    --- |ok|FAIL|PASS|ledger)|DRAIN-GROUP|never released|still open|A10|A9|skip|SKIP|panic|_test.go'
  echo "== $label exit ${PIPESTATUS[0]} $(date '+%H:%M:%S')"
}
one AllocFloor GOINFER_DRAIN_GROUP=1 -- '^TestAllocFloor$'
one A10ReportingGap GOINFER_DRAIN_GROUP=1 -- '^TestA10ReportingGap$'
one A10FloorIsPerProcess GOINFER_DRAIN_GROUP=1 -- '^TestA10FloorIsPerProcessOrPerDevice$'
one A10FloorIsPerProcess-third GOINFER_DRAIN_GROUP=1 GOINFER_A10_MODE=third -- '^TestA10FloorIsPerProcessOrPerDevice$'
one A10-drain-child-direct GOINFER_A10_DRAIN_CHILD=1 -- '^TestA10FloorIsPerProcessOrPerDevice$'
one A10CtxHolder-direct GOINFER_A10_CTX_HOLDER=1 -- '^TestA10CtxHolder$'
one MoERouteDemandThreshold GOINFER_DRAIN_GROUP=1 -- '^TestMoERouteDemandThreshold$'
one MoERouteDemandThresholdChild-direct GOINFER_A9_LEAVE_FREE=1073741824 -- '^TestMoERouteDemandThresholdChild$'
LOCK=" " one KSA1_gemvTiming GOINFER_KSA1_TIMING=1 -- '^TestKSA1_gemvTiming$'
