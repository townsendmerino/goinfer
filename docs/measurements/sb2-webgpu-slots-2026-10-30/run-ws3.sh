#!/bin/bash
# G-WS3 (SB2 memory reads step 4): the Vulkan card check. Run from the repo root; takes the timing lock exclusively.
# usage: bash docs/measurements/sb2-webgpu-slots-2026-10-30/run-ws3.sh <pass-name>
# Exit 75 (VOID) when anything but the desktop's own graphics clients is on the card before a measurement; any failed step stops it.
set -euo pipefail
export PATH=/usr/local/go/bin:$PATH
D=docs/measurements/sb2-webgpu-slots-2026-10-30
pass=${1:-pass1}
exec python3 scripts/timing_lock.py run --label "nobara-sb01 SB2 G-WS3 Vulkan check (~5 min)" -- bash -c '
set -euo pipefail
D='"$D"'; pass='"$pass"'
# The desktop shell'"'"'s own GPU clients, as nvidia-smi listed them on 2026-10-30 after the reboot.
desktop="kwin_wayland plasma-keyboard Xwayland plasmashell xwaylandvideobridge"
idle_or_void() { # $1: the measurement about to start
  local bad=""
  # Every GPU client nvidia-smi knows, graphics (Vulkan, GL) and compute alike.
  while read -r pid name; do
    case " $desktop " in *" $(basename "$name") "*) ;; *) bad="$bad $pid:$name";; esac
  done < <(nvidia-smi | sed -n "/Processes:/,\$p" | awk "/^\\|[[:space:]]+[0-9]/ {print \$5, \$7}")
  # And any process outside the desktop that has the NVIDIA Vulkan ICD or the CUDA driver mapped.
  for p in /proc/[0-9]*; do
    grep -qsE "libGLX_nvidia|libcuda\\.so" "$p/maps" 2>/dev/null || continue
    c=$(cat "$p/comm" 2>/dev/null || true)
    case " $desktop xwaylandvideobr " in *" $c "*) ;; *) bad="$bad $(basename "$p"):$c";; esac
  done
  if [ -n "$bad" ]; then echo "VOID before $1: on the card:$bad"; exit 75; fi
}
cp $D/sb2_ws3_fill_test.go.txt gpu/sb2_ws3_fill_test.go
cp $D/sb2_ws3_cudafree_test.go.txt cuda/sb2_ws3_cudafree_test.go
trap "rm -f gpu/sb2_ws3_fill_test.go cuda/sb2_ws3_cudafree_test.go" EXIT
# Build the test binaries first, so no compile runs between a reading and its load.
(cd cuda && go test -c -tags cuda -o ../$D/.ws3-cuda.test .) && (cd gpu && go test -c -tags "gpu goinfer_testhooks" -o ../$D/.ws3-gpu.test .)
trap "rm -f gpu/sb2_ws3_fill_test.go cuda/sb2_ws3_cudafree_test.go $D/.ws3-cuda.test $D/.ws3-gpu.test" EXIT
for m in qwen2.5-coder-1.5b-instruct-q4_k_m.gguf qwen2.5-7b-instruct-q4_k_m.gguf; do
  stem=$D/$pass-${m%%-instruct*}
  idle_or_void "C ($m)"
  { date "+%F %T %Z"; uptime; grep MemAvailable /proc/meminfo; nvidia-smi --query-gpu=memory.used,memory.free,driver_version --format=csv,noheader; nvidia-smi | sed -n "/Processes:/,\$p"; } > $stem-card.txt
  (cd cuda && SB2_WS3_OUT=$PWD/../$stem-C.txt ../$D/.ws3-cuda.test -test.run "^TestSB2WS3CudaFree\$" -test.count=1 -test.v) > $stem-C.log 2>&1
  idle_or_void "the fill ($m)"
  (cd gpu && SB2_WS3_MODEL=$HOME/models/$m SB2_WS3_OUT=$PWD/../$stem-fill.txt ../$D/.ws3-gpu.test -test.run "^TestSB2WS3Fill\$" -test.count=1 -test.v -test.timeout 10m) > $stem-fill.log 2>&1
  echo "== $m"; cat $stem-C.txt $stem-fill.txt
done
idle_or_void "the slot tests"
(cd gpu && GOINFER_HEAVY_TESTS=1 ../$D/.ws3-gpu.test -test.run "^TestWebGPUKVSlots_" -test.count=1 -test.v -test.timeout 15m) > $D/$pass-slots.log 2>&1
grep -E -- "^--- |granted|did not fit|MiB left" $D/$pass-slots.log
'
