#!/bin/bash
# ref-profile 2026-10-23: one CR7B reference prompt, profiled. Args: K STEPS WORKERS BUDGET TAG [MODEL QUANT]
set -u
export PATH=/usr/local/go/bin:$PATH
W=/home/francis/goinfer-bench/ref-profile-2026-10-23
O=/home/francis/goinfer-bench/ref-profile-2026-10-23-out
K=$1 STEPS=$2 WORKERS=$3 BUDGET=$4 TAG=$5 MODEL=${6:-$HOME/models/command-r7b} QUANT=${7-int8}
cd $W
[ -x $O/decoder.test ] || go test -c -tags goinfer_testhooks -o $O/decoder.test ./decoder/ || exit 1
echo "start $(date '+%F %T %Z') tree $(git rev-parse HEAD) load: $(cat /proc/loadavg)"; free -m | sed -n 2p
cd $W/decoder
GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_PREFILL_REF_CACHE=$O/ref-cache RP_MODEL=$MODEL RP_QUANT=$QUANT RP_K=$K RP_STEPS=$STEPS \
  RP_WORKERS=$WORKERS RP_BUDGET=$BUDGET RP_OUT=$O/$TAG \
  $O/decoder.test -test.run '^TestScratchRefProfile$' -test.v -test.timeout 20m
echo "end $(date '+%F %T %Z') load: $(cat /proc/loadavg)"
