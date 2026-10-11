#!/bin/bash
# The cold-load experiment, third run (exploratory): the shipped loader with and without a front-to-back read first,
# on the 35B MoE, the Mellum MoE and a dense 7B. Estimate: 10 min.
set -uo pipefail
D=$HOME/goinfer-bench/coldload; M=$HOME/models
echo "started $(date '+%Y-%m-%d %H:%M:%S %Z'); $(uptime | sed 's/.*load/load/')"
$HOME/goinfer-bench/v16-rebuild/prequant-v16-linux -quant int4 -target cpu-amd64 -o $D/out/7b.giw $M/qwen2.5-7b-instruct-q4_k_m.gguf 2>&1 | tail -1
bash $D/cold-c.sh $M/qwen3.6-35b-a3b-int4.giw 2
bash $D/cold-c.sh $M/mellum2.int4.giw 2
bash $D/cold-c.sh $D/out/7b.giw 2
find $D/out -name '7b.giw*' -delete
echo "finished $(date '+%H:%M:%S')"
