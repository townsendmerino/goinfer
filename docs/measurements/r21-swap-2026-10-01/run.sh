#!/usr/bin/env bash
D=$HOME/goinfer-bench/r21-swap-2026-10-01
S=$D/samples.tsv; : > $S
( while true; do awk -v t=$(date +%s) '/^SwapTotal/{st=$2}/^SwapFree/{sf=$2}/^MemAvailable/{ma=$2}/^Cached:/{c=$2}END{printf "%s\t%d\t%d\t%d\n",t,st-sf,ma,c}' /proc/meminfo >> $S; sleep 2; done ) &
SP=$!
mark() { echo "$(date +%s) $*" >> $D/marks.txt; echo "[$(date +%T)] $*"; }
: > $D/marks.txt
mark idle1-start; sleep 30; mark idle1-end
mark B-start; cat $HOME/models/gemma4-26b-q4_k_m.gguf > /dev/null; mark B-end
mark idle2-start; sleep 30; mark idle2-end
mark A-start
"$D/serve" -model $HOME/models/gemma4-26b-gguf/gemma-4-26B_q4_0-it.gguf -backend cuda -addr 127.0.0.1:8097 > $D/serve-A.log 2>&1 &
SV=$!
for i in $(seq 1 150); do grep -q "decode path:" $D/serve-A.log && break; kill -0 $SV 2>/dev/null || break; sleep 2; done
mark A-loaded; sleep 30; mark A-end
kill $SV; wait $SV 2>/dev/null
mark idle3-start; sleep 30; mark idle3-end
kill $SP
echo ALLDONE
