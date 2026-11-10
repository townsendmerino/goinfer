#!/usr/bin/env bash
# #873 corrected recipe: chat fit over every ~/models entry at both heads, one output file per entry per head, rc per entry.
# A = 589c6ebd9 (#873's merge-base), B = 55128eca3. Per entry A then B, so B's reads come from the page cache.
R=$HOME/goinfer-bench/873/run; cd $R
one() { d=${1%/}; n=$(printf '%s' "$d" | tr '/ ' '__')
  for X in A B; do timeout 900 $R/$X/chat fit "$d" > $R/out/$X/$n.txt 2>&1; echo "$? $d" > $R/out/$X/$n.rc; done; }
export -f one; export R
: > $R/list-srv.txt; : > $R/list-local.txt
for d in ~/models/*/ ~/models/*.gguf ~/models/*/*.gguf; do [ -e "$d" ] || continue
  case "$(readlink -f "${d%/}")" in /srv/models*) echo "$d" >> $R/list-srv.txt;; *) echo "$d" >> $R/list-local.txt;; esac; done
total=$(cat $R/list-srv.txt $R/list-local.txt | wc -l)
echo "$(date +%T) sweep start: $total entries ($(wc -l < $R/list-srv.txt) archive-backed, serial; $(wc -l < $R/list-local.txt) local, 4 parallel)"
( while :; do sleep 60; echo "$(date +%T) done $(ls $R/out/B/*.rc 2>/dev/null | wc -l) of $total"; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
xargs -a $R/list-srv.txt -d '\n' -n1 -P1 bash -c 'one "$0"' &
xargs -a $R/list-local.txt -d '\n' -n1 -P4 bash -c 'one "$0"' &
wait %2 %3 2>/dev/null; wait
for X in A B; do cat $R/out/$X/*.rc | sort -k2 > $R/rc-$X.txt; done
echo "$(date +%T) SWEEP-DONE done $(ls $R/out/B/*.rc | wc -l) of $total"
