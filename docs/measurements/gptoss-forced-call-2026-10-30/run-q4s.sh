#!/usr/bin/env bash
# The int4 quantizer arms (README.md, "P8 addendum: the int4 quantizers"). Detached, nobody watching. Builds the
# instrument from P8's source snapshot (~/goinfer-bench/p8-bin/src, e96ceb35e) with only the test file replaced, so the
# decoder and aikit code are P4g's. It waits for nobara's timing lock to be free, then runs fullrange and then mse, each
# under the exclusive lock. An arm not started by 2026-10-31 23:59 is not run. A void start retries in 10 minutes, a lock
# refusal in 2.
set -u
B=~/goinfer-bench/q4s-bin; SRC=~/goinfer-bench/p8-bin/src; S=~/goinfer-logs/mm01/gptoss-split
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
log() { echo "$(date '+%F %T') $*" | tee -a $S/q4s.log; }
late() { [ "$(date +%Y%m%d%H%M)" -gt 202610312359 ]; }
tl() { (cd $SRC && python3 scripts/timing_lock.py "$@"); }

log "waiting for the timing lock (no arm starts after 2026-10-31 23:59)"
until [ -x $B/namelogits.test ]; do
  late && { log "STOP: past 23:59 before the build; nothing run"; exit 9; }
  if [[ "$(tl status 2>&1 | head -1)" == free* ]]; then
    tl run --shared --wait 120 --label 'nobara-mm01 int4 quantizer arms: build the instrument (~2 min)' -- \
      bash -c "cp $B/namelogits_test.go.txt $SRC/cuda/zz_namelogits_test.go && cd $SRC/cuda && go test -c -tags 'cuda goinfer_testhooks' -o $B/namelogits.test.tmp . && mv $B/namelogits.test.tmp $B/namelogits.test" >> $S/q4s-build.log 2>&1
    log "build exit $?"
  fi
  [ -x $B/namelogits.test ] || sleep 120
done
log "instrument sha256 $(sha256sum $B/namelogits.test | cut -c1-12)"

one() { # the arm body, run under the exclusive lock: the void check, the instrument, the grade
  local a=$1 load avail p
  load=$(cut -d' ' -f1 /proc/loadavg); avail=$(awk '/^MemAvailable:/{printf "%.1f", $2*1024/1e9}' /proc/meminfo)
  log "$a start: load $load, MemAvailable $avail GB (need 13 + 2)"
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 8.0 else 1)" "$load" || { log "VOID $a: load $load > 8.0"; return 3; }
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 15 else 1)" "$avail" || { log "VOID $a: MemAvailable $avail GB < 13 + 2"; return 3; }
  NAMELOGITS_ARM=$a NAMELOGITS_DIR=$S $B/namelogits.test -test.run '^TestScratchGptOssNameLogits$' -test.v -test.count=1 -test.timeout 120m > $S/$a.log 2>&1
  p=$?; log "$a exit $p"; grep -E 'fit|refus|loaded in|steps in|^(--- |ok|FAIL|panic)' $S/$a.log | tee -a $S/q4s.log
  return $p
}
if [ "${1:-}" = "--one" ]; then one "$2"; exit $?; fi

arm() {
  local a=$1 rc
  while :; do
    late && { log "STOP: past 23:59; $a not started"; return 9; }
    tl run --wait 1800 --label "nobara-mm01 int4 quantizer arm $a (~40 min, timeout 120)" -- bash $B/run-q4s.sh --one "$a"
    rc=$?
    case $rc in
      3) log "$a void at start; retry in 10 min"; sleep 600 ;;
      75) log "$a: lock refused; retry in 2 min"; sleep 120 ;;
      *) log "$a done, exit $rc"; return $rc ;;
    esac
  done
}

arm cpu-int4g32-fullrange
arm cpu-int4g32-mse
python3 -I $B/compare_q4s.py $S 2>&1 | tee -a $S/q4s.log
log "end"
