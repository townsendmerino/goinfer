#!/bin/bash
# G-VR1 and G-VR3 (SB2-vk, M1′ restated): CPU only, eleven real checkpoints from ~/models; the build and the run hold the timing
# lock shared. usage, from the repo root: bash docs/measurements/sb2-vk-2026-10-31/run-vr1.sh
set -euo pipefail
export PATH=/usr/local/go/bin:$PATH
D=docs/measurements/sb2-vk-2026-10-31
cp $D/sb2_vr1_test.go.txt gpu/sb2_vr1_test.go
trap "rm -f gpu/sb2_vr1_test.go $D/.vr1-gpu.test" EXIT
python3 scripts/timing_lock.py run --shared --label "nobara-sb01 SB2-vk G-VR1 (CPU, ~10 min)" -- bash -c "
set -e
cd gpu && go test -c -tags 'gpu goinfer_testhooks' -o ../$D/.vr1-gpu.test .
CUDA_VISIBLE_DEVICES=-1 VK_ICD_FILENAMES=/dev/null SB2_VR1=1 SB2_VR1_OUT=\$PWD/../$D/vr1.txt ../$D/.vr1-gpu.test -test.run '^TestSB2VR1\$' -test.count=1 -test.v -test.timeout 30m" > $D/vr1.log 2>&1
