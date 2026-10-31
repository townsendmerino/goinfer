#!/bin/bash
# G-VK1 (SB2-vk): six WebGPU loads, each with its ledger's buffer list, nvidia-smi before/after the load and after a fill, and a cuda
# probe held open so nvidia-smi reads the CUDA context's size. Run from the repo root; takes the timing lock exclusively.
# usage: bash docs/measurements/sb2-vk-2026-10-31/run-vk1.sh <pass-name>
# Exit 75 (VOID) when anything but the desktop's own graphics clients is on the card before a measurement; any failed step stops it.
set -euo pipefail
export PATH=/usr/local/go/bin:$PATH
D=docs/measurements/sb2-vk-2026-10-31
pass=${1:-pass1}
exec python3 scripts/timing_lock.py run --label "nobara-sb01 SB2-vk G-VK1 (~10 min)" -- bash -c '
set -euo pipefail
D='"$D"'; pass='"$pass"'
desktop="kwin_wayland plasma-keyboard Xwayland plasmashell xwaylandvideobridge"
idle_or_void() { # $1: the measurement about to start
  local bad=""
  while read -r pid name; do
    case " $desktop " in *" $(basename "$name") "*) ;; *) bad="$bad $pid:$name";; esac
  done < <(nvidia-smi | sed -n "/Processes:/,\$p" | awk "/^\\|[[:space:]]+[0-9]/ {print \$5, \$7}")
  for p in /proc/[0-9]*; do
    grep -qsE "libGLX_nvidia|libcuda\\.so" "$p/maps" 2>/dev/null || continue
    c=$(cat "$p/comm" 2>/dev/null || true)
    case " $desktop xwaylandvideobr " in *" $c "*) ;; *) bad="$bad $(basename "$p"):$c";; esac
  done
  if [ -n "$bad" ]; then echo "VOID before $1: on the card:$bad"; exit 75; fi
}
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
cp $D/sb2_vk1_probe_test.go.txt gpu/sb2_vk1_probe_test.go
cp $D/sb2_vk1_cudactx_test.go.txt cuda/sb2_vk1_cudactx_test.go
trap "rm -f gpu/sb2_vk1_probe_test.go cuda/sb2_vk1_cudactx_test.go" EXIT
(cd cuda && go test -c -tags cuda -o ../$D/.vk1-cuda.test .) && (cd gpu && go test -c -tags "gpu goinfer_testhooks" -o ../$D/.vk1-gpu.test .)
trap "rm -f gpu/sb2_vk1_probe_test.go cuda/sb2_vk1_cudactx_test.go $D/.vk1-cuda.test $D/.vk1-gpu.test" EXIT
run() { # $1 model file, $2 tag, $3 ctx (empty for the default)
  local m=$1 stem=$D/$pass-$2
  idle_or_void "C ($2)"
  { date "+%F %T %Z"; uptime; grep MemAvailable /proc/meminfo; nvidia-smi --query-gpu=memory.used,memory.free,driver_version --format=csv,noheader; nvidia-smi | sed -n "/Processes:/,\$p"; } > $stem-card.txt
  local u0; u0=$(used)
  rm -f $stem-C.txt
  (cd cuda && SB2_VK1_OUT=$PWD/../$stem-C.txt SB2_VK1_HOLD=3 ../$D/.vk1-cuda.test -test.run "^TestSB2VK1CudaCtx\$" -test.count=1 -test.v) > $stem-C.log 2>&1 &
  local cp=$!
  for i in $(seq 1 100); do [ -s $stem-C.txt ] && break; sleep 0.1; done
  sleep 0.5
  local u1; u1=$(used)
  wait $cp
  echo "cuda_ctx nvsmi_used_MiB before=$u0 during=$u1" >> $stem-C.txt
  idle_or_void "the load ($2)"
  (cd gpu && SB2_VK1_MODEL=$HOME/models/$m SB2_VK1_CTX=$3 SB2_VK1_OUT=$PWD/../$stem-probe.txt ../$D/.vk1-gpu.test -test.run "^TestSB2VK1Probe\$" -test.count=1 -test.v -test.timeout 10m) > $stem-probe.log 2>&1
  echo "== $2"; cat $stem-C.txt $stem-probe.txt
}
run qwen2.5-coder-0.5b-instruct-q4_k_m.gguf qwen0.5b ""
run qwen2.5-coder-1.5b-instruct-q4_k_m.gguf qwen1.5b ""
run llama-3.2-1b-instruct-q4_k_m.gguf llama1b ""
run mistral-7b-instruct-v0.1.Q4_K_M.gguf mistral7b ""
run qwen2.5-7b-instruct-q4_k_m.gguf qwen7b ""
run qwen2.5-coder-1.5b-instruct-q4_k_m.gguf qwen1.5b-ctx4096 4096
'
