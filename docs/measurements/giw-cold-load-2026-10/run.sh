#!/bin/bash
# The cold-load experiment (exploratory): a dense v16 control is transcoded first, then part A on two MoE bundles and
# the control, then part B on the 35B. Estimate: 20 min.
set -uo pipefail
D=$HOME/goinfer-bench/coldload; M=$HOME/models
echo "started $(date '+%Y-%m-%d %H:%M:%S %Z'); $(uptime | sed 's/.*load/load/'); $(findmnt -no FSTYPE /home); read_ahead_kb: $(cat /sys/class/bdi/$(mountpoint -d /home)/read_ahead_kb 2>/dev/null || echo unknown)"
$HOME/goinfer-bench/v16-rebuild/prequant-v16-linux -quant int4 -target cpu-amd64 -o $D/out/7b.giw $M/qwen2.5-7b-instruct-q4_k_m.gguf 2>&1 | tail -1
bash $D/cold-a.sh $M/qwen3.6-35b-a3b-int4.giw $M/mellum2.int4.giw $D/out/7b.giw
bash $D/cold-b.sh $M/qwen3.6-35b-a3b-int4.giw
find $D/out -name '7b.giw*' -delete
echo "finished $(date '+%H:%M:%S')"
