#!/bin/bash
# G-Q2 and G-Q3 of docs/tasks/task-metal-q4k-2026-10.md on the MacBook, night queue (the fit guard refuses Phi-3 at q4k by
# day; at night the free memory is the budget). The load-time fit guard stays ON unless GQ_NO_FIT_GUARD=1, the owner's
# word for one run: both gates then load with GOINFER_NO_FIT_GUARD=1 under a swap watch that kills the gate's test and
# serve processes when swap grows more than GQ_SWAP_KILL_MB (default 1024) over the gate's own baseline. Metal's
# resident memory guard is never bypassed here.
# Pinned: a detached worktree at REV with its OWN go.work (this machine's global GOWORK points at the moving main
# checkout), a test binary and a serve binary built from it. Checkpoints from ~/models.
set -u
REV=${GQ_REV:?set GQ_REV to the pinned commit}
B=$HOME/goinfer-bench/metal-q4k
OUT=$B/results-$(date +%F)
R=$HOME/tmcode/goinfer
WT=$B/wt-$REV
PHI3=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
mkdir -p "$OUT"
[ -d "$WT" ] || git -C "$R" worktree add --detach "$WT" "$REV" || exit 1
[ "$(git -C "$WT" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./metal\n)\n' > "$WT/go.work"
export GOWORK=$WT/go.work
[ -x "$B/metal-$REV.test" ] || (cd "$WT/metal" && go test -c -tags goinfer_testhooks -o "$B/metal-$REV.test" .) || exit 1
[ -x "$B/serve-metal-$REV" ] || (cd "$WT/metal" && CGO_ENABLED=0 go build -o "$B/serve-metal-$REV" ./cmd/serve) || exit 1
{ echo "rev $REV (test sha256 $(shasum -a 256 "$B/metal-$REV.test" | cut -c1-16), serve sha256 $(shasum -a 256 "$B/serve-metal-$REV" | cut -c1-16))"
  echo "phi3 $(shasum -a 256 "$PHI3" | cut -c1-16)"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1
  sysctl -n vm.swapusage; vm_stat | head -5; uptime; } | tee "$OUT/provenance.txt"
KILL_MB=${GQ_SWAP_KILL_MB:-1024}
swap_used() { sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M' | grep -oE '[0-9.]+'; }
# watch_swap <gate>: samples swap every 3 s (one line a minute to the gate's watch log); over the limit it kills this
# job's pinned test and serve binaries by their launch paths, which fails the gate.
watch_swap() {
  local base now n=0; base=$(swap_used)
  echo "$(date '+%T') swap watch ($1): baseline ${base} MB, kill at +${KILL_MB} MB" | tee -a "$OUT/swapwatch-$1.log"
  while sleep 3; do
    now=$(swap_used); n=$((n + 1))
    [ $((n % 20)) -eq 0 ] && echo "$(date '+%T') swap ${now} MB (+$(echo "$now - $base" | bc))" >> "$OUT/swapwatch-$1.log"
    if [ "$(echo "$now - $base > $KILL_MB" | bc -l)" = 1 ]; then
      echo "$(date '+%T') swap watch ($1): +$(echo "$now - $base" | bc) MB is over ${KILL_MB} MB: KILLING the gate's processes" | tee -a "$OUT/swapwatch-$1.log"
      pkill -9 -f "^$B/metal-$REV.test" ; pkill -9 -f "^$B/serve-metal-$REV"
      return
    fi
  done
}
WATCH=""
start_watch() { WATCH=""; [ -n "${GQ_NO_FIT_GUARD:-}" ] || return 0; watch_swap "$1" & WATCH=$!; }
stop_watch() { [ -n "$WATCH" ] && { kill "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null; }; WATCH=""; }
if [ -n "${GQ_NO_FIT_GUARD:-}" ]; then
  export GOINFER_NO_FIT_GUARD=1
  echo "the load-time fit guard is BYPASSED for this run (GQ_NO_FIT_GUARD), swap watch at +${KILL_MB} MB" | tee -a "$OUT/provenance.txt"
fi
echo "=== $(date -u +%FT%TZ) START G-Q2"
start_watch gq2
# A pinned test binary runs from its package directory (it resolves ../testdata from there).
(cd "$WT/metal" && GOINFER_HEAVY_TESTS=1 "$B/metal-$REV.test" -test.run '^TestQ4KLane_realPhi3NonInferiority$' -test.v -test.count=1 -test.timeout 40m) 2>&1 | tee "$OUT/gq2.log"
rc=${PIPESTATUS[0]}
stop_watch
echo "=== $(date -u +%FT%TZ) END G-Q2 rc=$rc; swap $(sysctl -n vm.swapusage)"
echo "=== $(date -u +%FT%TZ) START G-Q3"
start_watch gq3
(cd "$WT" && bash docs/measurements/metal-q4k-2026-10/run-gq3-served.sh "$B/serve-metal-$REV" "$OUT/gq3" "$PHI3") 2>&1 | tee "$OUT/gq3.log"
rc2=${PIPESTATUS[0]}
stop_watch
echo "=== $(date -u +%FT%TZ) END G-Q3 rc=$rc2; swap $(sysctl -n vm.swapusage)"
[ $rc -eq 0 ] && rc=$rc2
exit $rc
