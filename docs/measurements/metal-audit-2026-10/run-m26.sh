#!/usr/bin/env bash
# Night job, owner-approved 2026-10-01 to run LAST in the queue: T1.8 and T1.9 of docs/tasks/task-metal-audit-2026-10.md
# on the Gemma 4 26B (M26), pre-registered in that doc's "The M26 job" table before this ran. Three processes, each under
# scripts/swap_killwatch.sh (copied beside the binary):
#   1. T1.8:  8 slots, held at token 32 while vmmap and footprint are taken from OUTSIDE the process;
#   2. T1.9a: 8 slots, GOINFER_MOE_PROF_SPLIT=1;
#   3. T1.9b: 64 slots, last, and only if neither earlier run tripped the kill-watch (64 slots spiralled into swap on
#      2026-09-20, before the 2026-09-24 fork fix).
# Nothing forks from the M26 process: a fork with GPU-wired pages is what collapsed M26 on 2026-09-24.
#
# Runs a test binary built at REV with -tags goinfer_testhooks, so the tree may move before tonight:
#   (cd ~/tmcode/goinfer-metal-audit && go test -c -tags goinfer_testhooks \
#     -o ~/goinfer-bench/metal-audit-2026-10/metal-tagged-<REV>.test ./metal/)
# Queued with:
#   python3 scripts/night.py add metal-audit-m26 --est 15 --priority 90 --by "Claude, Metal audit M26 (owner-approved)" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-m26.sh
# Logs: ~/goinfer-logs/metal-audit-2026-10/m26/; RESULT lines collected in results.txt there.
set -uo pipefail
REV=f56b40ec
BASE=$HOME/goinfer-bench/metal-audit-2026-10
BIN=$BASE/metal-tagged-$REV.test
KW=$BASE/swap_killwatch.sh
LOG=$HOME/goinfer-logs/metal-audit-2026-10/m26
M26=$HOME/models/gemma4-26b-int4-v14st.metal.giw
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
[ -f "$KW" ] || { echo "missing kill-watch $KW"; exit 1; }
[ -f "$M26" ] || { echo "missing model $M26"; exit 1; }
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "model:    $M26 ($(stat -f %z "$M26") bytes)"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "swap:     $(sysctl -n vm.swapusage)"
} | tee "$LOG/provenance.txt"

step() { echo "== $1 — $(date '+%T')" | tee -a "$LOG/provenance.txt"; }

# run <name> <slots> <hold 0|1> [ENV=VALUE ...]: one probe process under the kill-watch. Returns 9 if the watch fired.
run() {
  local name=$1 slots=$2 hold=$3
  shift 3
  local holdf=""
  if [ "$hold" = 1 ]; then holdf=$LOG/$name.hold; rm -f "$holdf"; fi
  step "$name ($slots slots)"
  env GOINFER_METAL_AUDIT_M26=1 GOINFER_AUDIT_MODEL="$M26" GOINFER_AUDIT_SLOTS="$slots" GOINFER_AUDIT_HOLD="$holdf" "$@" \
    "$BIN" -test.v -test.count=1 -test.timeout 15m -test.run '^TestAuditM26_pagedProbe$' > "$LOG/$name.log" 2>&1 &
  local pid=$!
  KILL_DELTA_MB=1024 TICK_MB=80 TICK_N=2 POLL_S=1 bash "$KW" "$pid" "$LOG/$name.killwatch.log" > /dev/null &
  local kw=$!
  if [ -n "$holdf" ]; then
    for _ in $(seq 1 900); do
      [ -s "$holdf" ] && break
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if [ -s "$holdf" ]; then
      vmmap -summary "$pid" > "$LOG/$name.vmmap.txt" 2>&1
      footprint -f bytes "$pid" > "$LOG/$name.footprint.txt" 2>&1
      rm -f "$holdf"
    fi
  fi
  wait "$pid"
  local rc=$?
  wait "$kw" 2>/dev/null
  echo "$name rc=$rc; swap now $(sysctl -n vm.swapusage)" | tee -a "$LOG/provenance.txt"
  if grep -q 'KILLING' "$LOG/$name.killwatch.log" 2>/dev/null; then
    echo "$name: the kill-watch fired" | tee -a "$LOG/provenance.txt"
    return 9
  fi
  return 0
}

fired=0
run t1.8-8slots 8 1 || fired=1
sleep 30 # let the previous process's pages go before the next load
if [ "$fired" = 0 ]; then run t1.9a-8slots-split 8 0 GOINFER_MOE_PROF_SPLIT=1 || fired=1; sleep 30; fi
if [ "$fired" = 0 ]; then
  run t1.9b-64slots 64 0 || fired=1
else
  echo "t1.9b-64slots: SKIPPED, an earlier run tripped the kill-watch" | tee -a "$LOG/provenance.txt"
fi

grep -h 'RESULT\|HOLD at token 32\|built in' "$LOG"/t1.*.log | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
