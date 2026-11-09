#!/bin/bash
# SB2-vk held-out set 6 (README, "Held-out set 6"), on nobara from the repo root at the branch head under judgement:
#   bash docs/measurements/sb2-vk-2026-10-31/run-heldout6.sh
# For each registered load: fetch the checkpoint into a new directory under ~/models (outside the lock), take the timing lock
# exclusively for the card reading (exit 75, VOID, when anything but the desktop's own clients is on the card), then delete the
# checkpoint. Then the fourteen-load CPU re-run (vr5) under the lock held shared, and the score. Any failed step stops it.
set -euo pipefail
export PATH=/usr/local/go/bin:$PATH
D=docs/measurements/sb2-vk-2026-10-31
pass=heldout6
if [ "${1:-}" = card ]; then # card <model path> <tag> <ctx> <ssm knob or empty>; run under the exclusive lock
  m=$2; stem=$D/$pass-$3; ctx=$4; ssm=$5
  desktop="kwin_wayland plasma-keyboard Xwayland plasmashell xwaylandvideobridge"
  idle_or_void() {
    local bad=""
    while read -r pid name; do
      case " $desktop " in *" $(basename "$name") "*) ;; *) bad="$bad $pid:$name";; esac
    done < <(nvidia-smi | sed -n "/Processes:/,\$p" | awk '/^\|[[:space:]]+[0-9]/ {print $5, $7}')
    for p in /proc/[0-9]*; do
      grep -qsE "libGLX_nvidia|libcuda\.so" "$p/maps" 2>/dev/null || continue
      c=$(cat "$p/comm" 2>/dev/null || true)
      case " $desktop xwaylandvideobr " in *" $c "*) ;; *) bad="$bad $(basename "$p"):$c";; esac
    done
    if [ -n "$bad" ]; then echo "VOID before $1: on the card:$bad"; exit 75; fi
  }
  used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
  if [ ! -s $D/$pass-vulkaninfo-memory.txt ]; then
    idle_or_void vulkaninfo
    vulkaninfo 2>/dev/null | sed -n "/VkPhysicalDeviceMemoryProperties/,/^VkPhysicalDevice[A-Z]/p" > $D/$pass-vulkaninfo-memory.txt
  fi
  idle_or_void "C ($3)"
  { date "+%F %T %Z"; uptime; grep MemAvailable /proc/meminfo; nvidia-smi --query-gpu=memory.used,memory.free,driver_version --format=csv,noheader; nvidia-smi | sed -n "/Processes:/,\$p"; } > $stem-card.txt
  u0=$(used); rm -f $stem-C.txt
  (cd backend/cuda && SB2_VK1_OUT=$PWD/../../$stem-C.txt SB2_VK1_HOLD=3 ../../$D/.vk5-cuda.test -test.run '^TestSB2VK1CudaCtx$' -test.count=1 -test.v) > $stem-C.log 2>&1 &
  cp=$!
  for i in $(seq 1 100); do [ -s $stem-C.txt ] && break; sleep 0.1; done
  sleep 0.5; u1=$(used); wait $cp
  echo "cuda_ctx nvsmi_used_MiB before=$u0 during=$u1" >> $stem-C.txt
  idle_or_void "the load ($3)"
  if (cd backend/webgpu && SB2_VK1_SSM=$ssm SB2_VK1_MODEL=$m SB2_VK1_CTX=$ctx SB2_VK1_OUT=$PWD/../../$stem-probe.txt ../../$D/.vk5-gpu.test -test.run '^TestSB2VK5Probe$' -test.count=1 -test.v -test.timeout 15m) > $stem-probe.log 2>&1; then
    echo "== $3"; cat $stem-C.txt $stem-probe.txt
  else
    rm -f $stem-probe.txt
    echo "== $3: NOT SCORED (did not load resident; $stem-probe.log)"
  fi
  exit 0
fi

HF=$(command -v hf || command -v huggingface-cli) || { echo "no hf / huggingface-cli on PATH"; exit 2; }
cleanup() { rm -f backend/webgpu/sb2_vk5_probe_test.go backend/webgpu/sb2_vr5_test.go backend/cuda/sb2_vk1_cudactx_test.go $D/.vk5-cuda.test $D/.vk5-gpu.test; [ -n "${cur:-}" ] && rm -rf "$cur"; true; }
trap cleanup EXIT
cp $D/sb2_vk5_probe_test.go.txt backend/webgpu/sb2_vk5_probe_test.go
cp $D/sb2_vr5_test.go.txt backend/webgpu/sb2_vr5_test.go
sed 's|goinfer/cuda"|goinfer/backend/cuda"|' $D/sb2_vk1_cudactx_test.go.txt > backend/cuda/sb2_vk1_cudactx_test.go
python3 scripts/timing_lock.py run --shared --label "nobara SB2-vk held-out 6 build" -- bash -c "
(cd backend/cuda && go test -c -tags cuda -o ../../$D/.vk5-cuda.test .) && (cd backend/webgpu && go test -c -tags 'gpu goinfer_testhooks' -o ../../$D/.vk5-gpu.test .)"

# The registered loads, in order (README, "Held-out set 6"); no substitution after a load is read.
# kind tag ctx ssm source
loads=(
  "hf nemotronh-4b-ctx8192 8192 1 nvidia/Nemotron-H-4B-Instruct-128K"
  "hf qwen35-4b-ctx8192 8192 - Qwen/Qwen3.5-4B"
  "hf mellum-4b-ctx4096 4096 - JetBrains/Mellum-4b-base"
  "hf qwen3-4b-2507-ctx8192 8192 - Qwen/Qwen3-4B-Instruct-2507"
)
for L in "${loads[@]}"; do
  read -r kind tag ctx ssm src <<<"$L"
  [ "$ssm" = - ] && ssm=""
  avail=$(df --output=avail -B1G "$HOME/models" | tail -1)
  if [ "$avail" -lt 20 ]; then echo "stop: ${avail} GB free under ~/models, fewer than 20"; exit 3; fi
  cur=$HOME/models/sb2-heldout6-$tag
  mkdir "$cur" # a new, empty directory: fails if one is left over
  if [ "$kind" = hf ]; then
    "$HF" download "$src" --local-dir "$cur" > $D/$pass-$tag-fetch.log 2>&1
    model=$cur
  else
    cp "$src" "$cur/" && model=$cur/$(basename "$src")
  fi
  gb=$(du -s -B1G "$cur" | cut -f1)
  if [ "$gb" -gt 15 ]; then echo "stop: $tag is ${gb} GB on disk, over 15"; exit 3; fi
  python3 scripts/timing_lock.py run --label "nobara SB2-vk held-out 6 card: $tag (~5 min)" -- bash "$0" card "$model" "$tag" "$ctx" "$ssm"
  rm -rf "$cur"; cur=""
done

python3 scripts/timing_lock.py run --shared --label "nobara SB2-vk vr6 (CPU, ~15 min)" -- bash -c "
cd backend/webgpu && CUDA_VISIBLE_DEVICES=-1 VK_ICD_FILENAMES=/dev/null SB2_VR5=1 SB2_VR5_OUT=\$PWD/../../$D/vr6.txt ../../$D/.vk5-gpu.test -test.run '^TestSB2VR5\$' -test.count=1 -test.v -test.timeout 45m" > $D/vr6.log 2>&1 || true
(cd $D && python3 score_heldout5.py . heldout6) | tee $D/$pass-score.txt
grep -E "^G-V|INCOMPLETE" $D/vr6.txt || tail -5 $D/vr6.log
