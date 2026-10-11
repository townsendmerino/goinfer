#!/bin/bash
# E2: one transcode of the 30B directory per arm, with the output file's page-cache residency and the kernel's dirty
# and writeback counters sampled every 2 s, so the state the self-check starts from is on record.
# e2.sh <arm> <prequant binary>
set -uo pipefail
D=$HOME/goinfer-bench/selfcheck
arm=$1; bin=$2
out=$D/out/30b.$arm.giw; tmp=$D/out/30b.$arm.tmp.giw
mkdir -p "$D/out"; find "$D/out" -name "30b.$arm*" -delete
stop=$D/sampling.$arm; : > "$stop"
python3 "$D/sampler.py" --stop-file "$stop" --interval 2 "$tmp" "$out" > "$D/samples.$arm.log" 2>&1 &
spid=$!
trap 'rm -f "$stop"; kill $spid 2>/dev/null' EXIT
echo "E2 $arm: started $(date '+%H:%M:%S')"
/usr/bin/time -f 'elapsed %es, max RSS %M kB, major faults %F, fs inputs %I, fs outputs %O' "$bin" -quant int4 -target cpu-amd64 -o "$out" "$HOME/models/qwen3-vl-30b-a3b-instruct" 2>&1 | tail -4
rm -f "$stop"; wait $spid 2>/dev/null
echo "E2 $arm: finished $(date '+%H:%M:%S'); $(python3 "$D/cachestat.py" "$out")"
