#!/bin/bash
# G-MS1 and G-MS2 (docs/tasks/task-subsystems-2026-10.md, "The WebGPU multi-slot finding"). Run from the repo root with the three test
# binaries prebuilt (see the record). Takes the timing lock exclusively; exits 75 (VOID) when anything but the desktop shell's GPU clients
# is on the card before a run.
set -uo pipefail
D=docs/measurements/webgpu-realclamp-2026-10-30
T=${T:?set T to the directory holding the prebuilt binaries}
ROOT=$PWD
exec python3 scripts/timing_lock.py run --wait 120 --label "nobara-sb01 G-MS1/G-MS2 realClamp (~3 min)" -- bash -c '
set -uo pipefail
D='"$D"'; T='"$T"'; ROOT='"$ROOT"'
desktop="kwin_wayland plasma-keyboard Xwayland plasmashell xwaylandvideobridge"
idle_or_void() {
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
run() { # $1 name, $2 binary, $3 tree, $4 -test.run pattern
  idle_or_void "$1"
  { date "+%F %T %Z"; nvidia-smi --query-gpu=memory.used,driver_version --format=csv,noheader; } > $ROOT/$D/$1-card.txt
  (cd $3/gpu && GOINFER_HEAVY_TESTS=1 $2 -test.run "$4" -test.count=1 -test.v -test.timeout 5m) > $ROOT/$D/$1.log 2>&1
  echo "$1 rc=$?: $(grep -E -- "^(--- |    --- )" $ROOT/$D/$1.log | tr "\n" " ")"
}
run gms1-parent $T/gpu-b6397eee5-parent.test $HOME/goinfer-bench/nobara-sb01-b6397eee5-parent-wt "^TestWebGPUKVSlots_realClamp\$/qwen2.5-coder-1.5b"
run gms1-b6397eee5 $T/gpu-b6397eee5.test $HOME/goinfer-bench/nobara-sb01-b6397eee5-wt "^TestWebGPUKVSlots_realClamp\$/qwen2.5-coder-1.5b"
run gms2-fix $T/gpu-fix.test $ROOT "^TestWebGPUKVSlots_realClamp\$"
'
