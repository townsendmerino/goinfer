#!/usr/bin/env bash
# Q1 split, the CPU arm P (README.md, "Q1 split"): goinfer's CPU path on the same int4 load, exact attention. Run by night.py,
# which holds the timing lock. Pre-built: the instrument binary in ~/goinfer-bench/gptoss-split-bin (sha256 e30c230b0f6d).
set -u
B=~/goinfer-bench/gptoss-split-bin; S=~/goinfer-logs/mm01/gptoss-split
log() { echo "$(date '+%F %T') $*" | tee -a $S/cpu.log; }
load=$(cut -d' ' -f1 /proc/loadavg); avail=$(awk '/^MemAvailable:/{printf "%.1f", $2*1024/1e9}' /proc/meminfo)
log "P start: load $load, MemAvailable $avail GB"
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 8.0 else 1)" "$load" || { log "VOID P: load $load > 8.0"; exit 1; }
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 15 else 1)" "$avail" || { log "VOID P: MemAvailable $avail GB < 13 + 2"; exit 1; }
NAMELOGITS_ARM=cpu NAMELOGITS_DIR=$S $B/namelogits.test -test.run '^TestScratchGptOssNameLogits$' -test.v -test.count=1 -test.timeout 70m > $S/p.log 2>&1
p=$?; log "P exit $p"; grep -E 'loaded in|steps in|^(--- |ok|FAIL)' $S/p.log | tee -a $S/cpu.log
python3 $S/compare.py $S 2>&1 | tee -a $S/cpu.log
exit $p
