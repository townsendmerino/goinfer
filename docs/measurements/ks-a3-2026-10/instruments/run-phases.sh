#!/usr/bin/env bash
# KS-A3 phase read (docs/measurements/ks-a3-2026-10/README.md): builds the phase-stamped topk PTX, copies the test into cuda/,
# runs it under the timing lock, removes the copy. Run from the repo root of a worktree whose go.work covers ./cuda.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export PATH=/usr/local/go/bin:$PATH
OUT=${1:-$HOME/goinfer-logs/ks-a3-2026-10}
mkdir -p "$OUT"
NV=$(ls -d "$HOME"/.local/lib/python*/site-packages/nvidia | head -1)
cp docs/measurements/ks-a3-2026-10/instruments/topk_phases.cu.txt "$OUT/topk_phases.cu"
NVRTC_SO=$NV/cuda_nvrtc/lib/libnvrtc.so.12 LD_LIBRARY_PATH=$NV/cuda_nvrtc/lib \
  python3 cuda/nvrtc_compile.py "$OUT/topk_phases.cu" "$OUT/topk_phases.ptx" compute_75 "$NV/cuda_runtime/include"
cp docs/measurements/ks-a3-2026-10/instruments/ks_a3_phases_test.go.txt cuda/ks_a3_scratch_test.go
trap 'rm -f cuda/ks_a3_scratch_test.go' EXIT
(cd cuda && go test -c -tags 'cuda goinfer_testhooks' -o "$OUT/ks-a3.test" .)
rm -f cuda/ks_a3_scratch_test.go
cd cuda && GOINFER_KS=1 GOINFER_KS_A3_PTX="$OUT/topk_phases.ptx" python3 ../scripts/timing_lock.py run --wait 3600 --label cuda-builds-ks-a3 -- \
  "$OUT/ks-a3.test" -test.run '^TestKSA3_phases$' -test.v -test.timeout 10m 2>&1 | grep -v '^\[resident\]' | tee "$OUT/phases.log"
