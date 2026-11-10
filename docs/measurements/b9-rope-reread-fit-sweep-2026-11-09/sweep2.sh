#!/usr/bin/env bash
# #873 sweep, resumed strictly serial (the first run's parallel fits swapped 27 GB). Entries that already have a B .rc are kept.
# Watchdog per fit: kill it if system swap grows over 512 MB past its start, or MemAvailable falls under 4 GB; rc 137, line MEMKILL in the .rc.
R=$HOME/goinfer-bench/873/run; cd $R
swapused() { awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo; }
avail()    { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
fit() { X=$1; d=$2; n=$3; base=$(swapused)
  timeout 900 $R/$X/chat fit "$d" > $R/out/$X/$n.txt 2>&1 & p=$!
  while kill -0 $p 2>/dev/null; do
    if [ $(( $(swapused) - base )) -gt 512 ] || [ "$(avail)" -lt 4096 ]; then pkill -x chat; echo "MEMKILL swap+$(( $(swapused) - base )) MB, avail $(avail) MB" >> $R/out/$X/$n.txt; wait $p 2>/dev/null; return 137; fi
    sleep 2; done
  wait $p; }
total=$(cat $R/list-srv.txt $R/list-local.txt | wc -l); todo=0
for d in $(cat $R/list-srv.txt $R/list-local.txt | tr ' ' '\001'); do d=$(printf '%s' "$d" | tr '\001' ' '); d=${d%/}
  n=$(printf '%s' "$d" | tr '/ ' '__'); [ -e $R/out/B/$n.rc ] && continue; todo=$((todo+1)); done
echo "$(date +%T) resume: $todo of $total left, serial, watchdog on"
i=0; for d in $(cat $R/list-local.txt $R/list-srv.txt | tr ' ' '\001'); do d=$(printf '%s' "$d" | tr '\001' ' '); d=${d%/}
  n=$(printf '%s' "$d" | tr '/ ' '__'); [ -e $R/out/B/$n.rc ] && continue
  for X in A B; do fit $X "$d" "$n"; echo "$? $d" > $R/out/$X/$n.rc; done
  i=$((i+1)); echo "$(date +%T) done $(ls $R/out/B/*.rc | wc -l) of $total  ($d)"
done
for X in A B; do cat $R/out/$X/*.rc | sort -k2 > $R/rc-$X.txt; done
echo "$(date +%T) SWEEP-DONE done $(ls $R/out/B/*.rc | wc -l) of $total"
