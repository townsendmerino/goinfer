#!/bin/bash
# Kernel-survey day runs (docs/measurements/kernel-survey-2026-10/README.md). Usage: ks.sh <phase>
# Each phase is launched by the caller under timing_lock.py (exclusive for timed, --shared otherwise).
set -o pipefail
export PATH=/usr/local/go/bin:$PATH
WT=/home/francis/goinfer-bench/kernel-survey-2026-10-23
D=/home/francis/goinfer-logs/kernel-survey-2026-10
NCU=/opt/nvidia/nsight-compute/2026.1.1/ncu
cd $WT || exit 2
stamp() { echo "== $1 $(date -Is) head $(git rev-parse --short HEAD) load $(cut -d' ' -f1-3 /proc/loadavg) memavail $(awk '/MemAvailable/{print int($2/1024)" MB"}' /proc/meminfo)"; }
case "$1" in
build)
  stamp build
  cp docs/measurements/kernel-survey-2026-10/instruments/ks_survey_test.go.txt cuda/ks_survey_scratch_test.go
  gofmt -l cuda/ks_survey_scratch_test.go
  go vet -tags 'cuda goinfer_testhooks' ./cuda/ && go test -c -tags 'cuda goinfer_testhooks' -o $D/ks.test ./cuda/
  rc=$?; rm -f cuda/ks_survey_scratch_test.go; echo "== build rc=$rc"; exit $rc ;;
a2)
  stamp a2
  cd cuda && GOINFER_HEAVY_TESTS=1 GOINFER_KS=1 $D/ks.test -test.run '^(TestCUDA_launchCost|TestKS_A2_graphsAB)$' -test.v -test.timeout 15m ;;
b8)
  cd cuda
  for w in 512 1024 2048 2048 1024 512; do
    stamp "b8 width=$w"
    GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_CHUNK=$w GOINFER_DECOMP_K=2048,3900 $D/ks.test -test.run '^TestPrefillDecomp$' -test.v -test.timeout 15m || exit $?
  done ;;
a1)
  stamp a1
  cd cuda && GOINFER_HEAVY_TESTS=1 GOINFER_KS=1 $D/ks.test -test.run '^TestKS_A1_planBytes$' -test.v -test.timeout 15m ;;
ncu-a3a5)
  stamp ncu-a3a5
  cd cuda && GOINFER_HEAVY_TESTS=1 GOINFER_KS=1 $NCU -k 'regex:^(topk_select|fused_rms_gu)' \
    --section SpeedOfLight --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis \
    -o $D/ncu-a3a5-$2 -f $D/ks.test -test.run '^TestKS_A3A5_decodeUnderNcu$' -test.v -test.timeout 30m ;;
ncu-b2)
  # $2 = tag, $3 = launch-skip, $4 = count
  stamp "ncu-b2 $2"
  cd cuda && GOINFER_HEAVY_TESTS=1 GOINFER_KS=1 $NCU -k 'regex:attn_fused_bm128_hd128' --launch-skip ${3:-168} -c ${4:-4} \
    --section WarpStateStats --section SchedulerStats --section SpeedOfLight --section Occupancy --section LaunchStats \
    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,smsp__sass_inst_executed_op_global_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
    -o $D/ncu-b2-$2 -f $D/ks.test -test.run '^TestKS_B2_prefillUnderNcu$' -test.v -test.timeout 40m ;;
*) echo "unknown phase $1"; exit 2 ;;
esac
