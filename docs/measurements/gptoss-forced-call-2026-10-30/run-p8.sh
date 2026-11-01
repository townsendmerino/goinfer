#!/usr/bin/env bash
# P8 by day (README.md, "P8, registered"; the coordinator's approval of 2026-10-31). Detached, nobody watching. Waits for
# nobara's timing lock, builds the instrument from the pinned source snapshot in ~/goinfer-bench/p8-bin/src (shared lock),
# runs P8 under the exclusive lock, and runs P88 then P4g only on P8's QUANTIZATION verdict. No arm starts after 19:30; a
# void start is retried after 10 minutes, a lock refusal after 2.
set -u
B=~/goinfer-bench/p8-bin; S=~/goinfer-logs/mm01/gptoss-split; R=$B/src # the snapshot's timing_lock.py (it has --shared)
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
log() { echo "$(date '+%F %T') $*" | tee -a $S/p8.log; }
late() { [ "$(date +%Y%m%d%H%M)" -gt 202610311930 ]; }
tl() { (cd $R && python3 scripts/timing_lock.py "$@"); }

log "waiting for the timing lock (no arm starts after 19:30)"
until [ -x $B/namelogits.test ]; do
  late && { log "STOP: past 19:30 before the build; nothing run"; exit 9; }
  if [[ "$(tl status 2>&1 | head -1)" == free* ]]; then
    tl run --shared --wait 120 --label 'nobara-mm01 P8: build the instrument (~2 min)' -- \
      bash -c "cd $B/src/cuda && go test -c -tags 'cuda goinfer_testhooks' -o $B/namelogits.test.tmp . && mv $B/namelogits.test.tmp $B/namelogits.test" >> $S/p8-build.log 2>&1
    log "build exit $?"
  fi
  [ -x $B/namelogits.test ] || sleep 120
done
log "instrument sha256 $(sha256sum $B/namelogits.test | cut -c1-12)"

arm() { # arm need: run one arm under the exclusive lock, retrying a refusal or a void start until 19:30
  local a=$1 need=$2 rc
  while :; do
    late && { log "STOP: past 19:30; $a not started"; return 9; }
    tl run --wait 1800 --label "nobara-mm01 P8: $a (~35-60 min, timeout 120)" -- bash $B/run-p8-arm.sh "$a" "$need"
    rc=$?
    case $rc in
      3) log "$a void at start; retry in 10 min"; sleep 600 ;;
      75) log "$a: lock refused; retry in 2 min"; sleep 120 ;;
      *) log "$a done, exit $rc"; return $rc ;;
    esac
  done
}

arm cpu-int8 24
v=$(python3 -I $B/compare_p8.py $S 2>/dev/null | grep '^VERDICT P8')
log "$v"
if [[ $v == *QUANTIZATION* ]]; then
  arm cpu-int8int8 24
  arm cpu-int4g32 13
  python3 -I $B/compare_p8.py $S 2>&1 | grep '^VERDICT' | tee -a $S/p8.log
fi
log "end"
