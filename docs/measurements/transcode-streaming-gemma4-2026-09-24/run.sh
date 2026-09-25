#!/bin/bash
# S2 gemma4 streaming transcode: resident (HEAD) vs streaming, same GGUF, RSS sampled every 200 ms.
L=~/goinfer-logs/s2; SRC=/home/francis/models/gemma4-26b-q4_k_m.gguf
for arm in head stream; do
  OUT=/home/francis/models/s2-$arm.int4.metal.giw; rm -f $OUT $OUT.verified
  echo "== $arm start $(date +%T)" >> $L/progress.txt
  nice -n 19 ionice -c3 /usr/bin/time -v /tmp/prequant-s2-$arm -quant int4 -target metal -o $OUT $SRC > $L/$arm.log 2>&1 &
  sleep 0.3; P=$(pgrep -f "^/tmp/prequant-s2-$arm" | head -1)
  echo "t_s,vmrss_kb,rssanon_kb,rssfile_kb" > $L/$arm.rss.csv
  t0=$(date +%s.%N)
  while kill -0 $P 2>/dev/null; do
    a=$(awk "/^VmRSS/{r=\$2}/^RssAnon/{a=\$2}/^RssFile/{f=\$2}END{print r\",\"a\",\"f}" /proc/$P/status 2>/dev/null)
    [ -n "$a" ] && echo "$(echo "$(date +%s.%N) - $t0" | bc),$a" >> $L/$arm.rss.csv
    sleep 0.2
  done
  wait
  echo "== $arm done $(date +%T) $(grep -E "Elapsed|Maximum resident|Exit status" $L/$arm.log | tr "\n" " ")" >> $L/progress.txt
done
echo ALLDONE $(date +%T) >> $L/progress.txt
