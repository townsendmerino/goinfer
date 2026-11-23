#!/bin/bash
# Attention-prefetch Phase 0 (docs/measurements/attn-prefetch-2026-11/README.md). Run under
# `timing_lock.py run --label attn-prefetch-p0 -- bash ap0.sh` (exclusive): ncu's reads and the decomposition time the card.
set -o pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
WT=/home/francis/goinfer-bench/attnprefetch-wt
D=/home/francis/goinfer-logs/attn-prefetch-2026-11/p0
NCU=/opt/nvidia/nsight-compute/2026.1.1/ncu
mkdir -p $D
cd $WT || exit 2
stamp() { echo "== $1 $(date -Is) head $(git rev-parse --short HEAD) load $(cut -d' ' -f1-3 /proc/loadavg) memavail $(awk '/MemAvailable/{print int($2/1024)" MB"}' /proc/meminfo)"; nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm --format=csv,noheader; }
stamp build
cp docs/measurements/attn-prefetch-2026-11/instruments/ap0_test.go.txt backend/cuda/ap0_scratch_test.go
go vet -tags 'cuda goinfer_testhooks' ./backend/cuda/ && go test -c -tags 'cuda goinfer_testhooks' -o $D/ap0.test ./backend/cuda/
rc=$?; rm -f backend/cuda/ap0_scratch_test.go; echo "== build rc=$rc"; [ $rc = 0 ] || exit $rc
cd backend/cuda
for kv in f32 f16; do
  stamp "ncu-b2 $kv"
  GOINFER_HEAVY_TESTS=1 GOINFER_KS=1 GOINFER_AP_KV=$kv $NCU -k 'regex:attn_fused_bm128_hd128' --launch-skip 168 -c 4 \
    --section WarpStateStats --section SchedulerStats --section SpeedOfLight --section Occupancy --section LaunchStats \
    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,smsp__sass_inst_executed_op_global_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
    -o $D/ncu-b2-$kv -f $D/ap0.test -test.run '^TestKS_B2_prefillUnderNcu$' -test.v -test.timeout 20m || exit $?
done
for kv in f32 f16; do
  stamp "decomp $kv K=3900"
  GOINFER_HEAVY_TESTS=1 GOINFER_KS=1 GOINFER_AP_KV=$kv GOINFER_DECOMP_K=3900 $D/ap0.test -test.run '^TestAP_prefillDecompKV$' -test.v -test.timeout 10m || exit $?
  stamp "decomp $kv K=8192 ctx 8192"
  GOINFER_HEAVY_TESTS=1 GOINFER_KS=1 GOINFER_AP_KV=$kv GOINFER_AP_CTX=8192 GOINFER_DECOMP_K=3900,8192 $D/ap0.test -test.run '^TestAP_prefillDecompKV$' -test.v -test.timeout 10m || echo "== decomp $kv 8192 rc=$? (does not fit?)"
done
stamp done
