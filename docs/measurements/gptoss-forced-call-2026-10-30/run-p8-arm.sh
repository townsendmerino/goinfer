#!/usr/bin/env bash
# One P8 arm (README.md, "P8, registered"): goinfer's CPU path, exact attention, at one load precision. Run under nobara's
# exclusive timing lock by run-p8.sh. Usage: run-p8-arm.sh <arm: cpu-int8 | cpu-int8int8 | cpu-int4g32> <need GB>.
# Exits 3 on a void start (load or memory), else the instrument's exit status.
set -u
arm=$1; need=$2
B=~/goinfer-bench/p8-bin; S=~/goinfer-logs/mm01/gptoss-split
log() { echo "$(date '+%F %T') $*" | tee -a $S/p8.log; }
load=$(cut -d' ' -f1 /proc/loadavg); avail=$(awk '/^MemAvailable:/{printf "%.1f", $2*1024/1e9}' /proc/meminfo)
log "$arm start: load $load, MemAvailable $avail GB (need $need + 2)"
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 8.0 else 1)" "$load" || { log "VOID $arm: load $load > 8.0"; exit 3; }
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) + 2 else 1)" "$avail" "$need" || { log "VOID $arm: MemAvailable $avail GB < $need + 2"; exit 3; }
NAMELOGITS_ARM=$arm NAMELOGITS_DIR=$S $B/namelogits.test -test.run '^TestScratchGptOssNameLogits$' -test.v -test.count=1 -test.timeout 120m > $S/$arm.log 2>&1
p=$?; log "$arm exit $p"; grep -E 'fit|refus|loaded in|steps in|^(--- |ok|FAIL|panic)' $S/$arm.log | tee -a $S/p8.log
python3 -I $B/compare_p8.py $S 2>&1 | tee -a $S/p8.log
exit $p
