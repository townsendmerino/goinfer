#!/usr/bin/env bash
# #873 sweep, retry of the heads the watchdog killed (first attempts kept in first-attempt/). Same watchdog, serial, box quiet.
R=$HOME/goinfer-bench/873/run; cd $R
swapused() { awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo; }
avail()    { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
fit() { X=$1; d=$2; n=$3; base=$(swapused)
  timeout 900 $R/$X/chat fit "$d" > $R/out/$X/$n.txt 2>&1 & p=$!
  while kill -0 $p 2>/dev/null; do
    if [ $(( $(swapused) - base )) -gt 512 ] || [ "$(avail)" -lt 4096 ]; then pkill -x chat; echo "MEMKILL swap+$(( $(swapused) - base )) MB, avail $(avail) MB" >> $R/out/$X/$n.txt; wait $p 2>/dev/null; return 137; fi
    sleep 2; done
  wait $p; }
total=$(wc -l < $R/retry-list.txt); i=0
while read X n; do i=$((i+1)); d=$(sed 's#^_#/#;s#_#/#g' <<< "$n"); 
  d=$(awk -v n="$n" '{x=$0; sub(/\/$/,"",x); gsub(/[\/ ]/,"_",x); if (x==n) {print $0; exit}}' $R/list-srv.txt $R/list-local.txt); d=${d%/}
  fit $X "$d" "$n"; rc=$?; echo "$rc $d" > $R/out/$X/$n.rc
  echo "$(date +%T) retry $i of $total: $X $n rc=$rc $(grep -c MEMKILL $R/out/$X/$n.txt) memkill"
done < $R/retry-list.txt
for X in A B; do cat $R/out/$X/*.rc | sort -k2 > $R/rc-$X.txt; done
echo "$(date +%T) RETRY-DONE"
