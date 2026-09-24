#!/bin/bash
# External, out-of-process swap-growth kill switch for runs that load a checkpoint the machine
# cannot comfortably hold (S5/S6, docs/tasks/task-never-swap-2026-09.md). Independent of decoder's
# S3 in-process tripwire: R11(c)'s lesson was that an in-process RSS read can look fine while swap
# actually explodes, so this watches REAL swap-used via sysctl and kills on a real delta.
#
#   swap_killwatch.sh <pid> <logfile>
#
# Env (defaults in brackets):
#   KILL_DELTA_MB [2048]  kill when swap-used exceeds this run's own baseline by this much
#   POLL_S        [3]     seconds between samples
#   TICK_MB       [0]     R11(c)-style rate rule, OFF at 0: kill when swap grows by more than TICK_MB
#                         on TICK_N consecutive samples, or by more than 4*TICK_MB in a single sample
#   TICK_N        [2]
#   DISK_FLOOR_GB [2]     kill when free disk on / drops under this
set -u
PID="$1"; LOG="$2"
KILL_DELTA_MB="${KILL_DELTA_MB:-2048}"; POLL_S="${POLL_S:-3}"; TICK_MB="${TICK_MB:-0}"; TICK_N="${TICK_N:-2}"
DISK_FLOOR_GB="${DISK_FLOOR_GB:-2}"
swap() { sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M' | grep -oE '[0-9.]+'; }
BASE=$(swap); PREV=$BASE; STREAK=0; START=$(date +%s)
echo "$(date '+%H:%M:%S') killwatch: baseline swap-used ${BASE}MB; kill at +${KILL_DELTA_MB}MB, tick rule ${TICK_MB}MB x${TICK_N} (0=off), poll ${POLL_S}s, pid $PID" | tee -a "$LOG"
kill_it() { echo "$(date '+%H:%M:%S') killwatch: $1 -- KILLING $PID" | tee -a "$LOG"; kill -9 "$PID" 2>/dev/null; exit 1; }
while kill -0 "$PID" 2>/dev/null; do
  NOW=$(swap); RSS=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' '); FREE=$(df -g / | tail -1 | awk '{print $4}')
  DELTA=$(echo "$NOW - $BASE" | bc); TICK=$(echo "$NOW - $PREV" | bc); PREV=$NOW
  echo "$(date '+%H:%M:%S') t+$(( $(date +%s) - START ))s swap_used=${NOW}MB delta=+${DELTA}MB tick=${TICK}MB rss=${RSS}KB free_disk=${FREE}G" | tee -a "$LOG"
  [ "$(echo "$DELTA > $KILL_DELTA_MB" | bc -l)" = 1 ] && kill_it "swap delta +${DELTA}MB > ${KILL_DELTA_MB}MB"
  if [ "$(echo "$TICK_MB > 0" | bc -l)" = 1 ]; then
    [ "$(echo "$TICK > 4 * $TICK_MB" | bc -l)" = 1 ] && kill_it "single-tick swap jump +${TICK}MB > $(echo "4 * $TICK_MB" | bc)MB"
    if [ "$(echo "$TICK > $TICK_MB" | bc -l)" = 1 ]; then STREAK=$((STREAK+1)); else STREAK=0; fi
    [ "$STREAK" -ge "$TICK_N" ] && kill_it "swap grew >${TICK_MB}MB on $STREAK consecutive ticks"
  fi
  [ "$FREE" -lt "$DISK_FLOOR_GB" ] && kill_it "free disk ${FREE}G < ${DISK_FLOOR_GB}G"
  sleep "$POLL_S"
done
echo "$(date '+%H:%M:%S') killwatch: pid $PID exited on its own, stopping watch" | tee -a "$LOG"
