#!/usr/bin/env bash
# SB12-I1 green, at this branch's tip: cuda-parity in full, cuda-heavy-small in full, then the three heavy groups whose I1 breaks
# the 2026-10-27 night printed (their -run lists are that night's top-level tests), each as gate gpu runs its cell. A green
# step prints no LEDGER BREAK and no I1 line. Each step starts when no other CUDA process is on the card, and a sampler logs any
# that appears while it runs (green-<step>.gpu): such a step is void under SB12-I1.
set -u
cd "$(dirname "$0")/../../.."
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
D=docs/measurements/sb12-i1i2-2026-10-29
unset GOINFER_HEAVY_TESTS GOINFER_DRAIN_GROUP
# CUDA processes other than the desktop's and this worktree's own test binary (told apart by its working directory).
others() {
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xwayland|plasmashell" |
    while IFS=', ' read -r pid name; do
      cwd=$(readlink /proc/$pid/cwd 2>/dev/null)
      case "$cwd" in "$PWD"/*) ;; *) echo "$pid $name cwd=${cwd:-?}" ;; esac
    done
}
gpu_free() {
  while :; do
    other=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xwayland|plasmashell")
    [ -z "$other" ] && return
    echo "   waiting $(date '+%H:%M:%S'): other CUDA process(es): $other"
    sleep 60
  done
}
step() { # name, run regex file or "", heavy (0/1), timeout
  local name=$1 run=$2 heavy=$3 to=$4
  gpu_free
  echo "== $name start $(date '+%F %H:%M:%S %Z') $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
  : > $D/green-$name.gpu
  (while :; do o=$(others); [ -n "$o" ] && echo "$(date '+%H:%M:%S') $o" >> $D/green-$name.gpu; sleep 60; done) & local s=$!
  local envs=() args=()
  [ "$heavy" = 1 ] && envs=(GOINFER_HEAVY_TESTS=1 GOINFER_ISOLATE_DEFER=1)
  [ -n "$run" ] && args=(-run "$(cat $run)")
  env "${envs[@]}" python3 scripts/timing_lock.py run --shared --wait 3600 --label "i1i2-green-$name" -- \
    go test -tags cuda,goinfer_testhooks -count=1 -v -timeout "$to" "${args[@]}" ./cuda/ > $D/green-$name.log 2>&1
  local rc=$?
  kill $s 2>/dev/null
  echo "== $name exit $rc $(date '+%F %H:%M:%S %Z'); other CUDA processes seen during it: $(wc -l < $D/green-$name.gpu)"
}
step parity "" 0 20m
step heavy-small docs/measurements/sb12-ledger-guard-2026-10-28/heavy-small.run 1 120m
for g in qwen25vl-3b 7b-8b qwen3-4b; do step $g $D/heavy-$g.run 1 120m; done
