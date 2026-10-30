#!/bin/bash
# G-WS3 (SB2 memory reads step 4): the Vulkan card check. Run from the repo root with the card otherwise idle; takes the timing lock
# exclusively. usage: bash docs/measurements/sb2-webgpu-slots-2026-10-30/run-ws3.sh <pass-name>
set -u
export PATH=/usr/local/go/bin:$PATH
D=docs/measurements/sb2-webgpu-slots-2026-10-30
pass=${1:-pass1}
exec python3 scripts/timing_lock.py run --label "nobara-sb01 SB2 G-WS3 Vulkan check (~5 min)" -- bash -c '
set -u
D='"$D"'; pass='"$pass"'
others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v kwin_wayland)
if [ -n "$others" ]; then echo "VOID: another compute process on the card: $others"; exit 75; fi
cp $D/sb2_ws3_fill_test.go.txt gpu/sb2_ws3_fill_test.go
cp $D/sb2_ws3_cudafree_test.go.txt cuda/sb2_ws3_cudafree_test.go
trap "rm -f gpu/sb2_ws3_fill_test.go cuda/sb2_ws3_cudafree_test.go" EXIT
for m in qwen2.5-coder-1.5b-instruct-q4_k_m.gguf qwen2.5-7b-instruct-q4_k_m.gguf; do
  stem=$D/$pass-${m%%-instruct*}
  { date "+%F %T %Z"; uptime; grep MemAvailable /proc/meminfo; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; nvidia-smi --query-gpu=memory.used,memory.free,driver_version --format=csv,noheader; } > $stem-card.txt
  (cd cuda && SB2_WS3_OUT=$PWD/../$stem-C.txt go test -tags cuda -run "^TestSB2WS3CudaFree\$" -count=1 -v .) > $stem-C.log 2>&1
  (cd gpu && SB2_WS3_MODEL=$HOME/models/$m SB2_WS3_OUT=$PWD/../$stem-fill.txt go test -tags "gpu goinfer_testhooks" -run "^TestSB2WS3Fill\$" -count=1 -v -timeout 10m .) > $stem-fill.log 2>&1
  echo "== $m"; cat $stem-C.txt $stem-fill.txt
done
(cd gpu && GOINFER_HEAVY_TESTS=1 go test -tags "gpu goinfer_testhooks" -run "^TestWebGPUKVSlots_" -count=1 -v -timeout 15m .) > $D/$pass-slots.log 2>&1
echo "slot tests rc=$?"; grep -E -- "^--- |granted|did not fit|MiB left" $D/$pass-slots.log
'
