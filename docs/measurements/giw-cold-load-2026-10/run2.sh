#!/bin/bash
# The cold-load experiment, second run (exploratory): a whole-file prefetch (MADV_WILLNEED) before the header walk,
# alone and with the walk in random mode, against the default in the same run. Estimate: 8 min.
set -uo pipefail
D=$HOME/goinfer-bench/coldload; M=$HOME/models
echo "started $(date '+%Y-%m-%d %H:%M:%S %Z'); $(uptime | sed 's/.*load/load/')"
MODES="default willneed willneed,random" bash $D/cold-a.sh $M/qwen3.6-35b-a3b-int4.giw $M/mellum2.int4.giw
MODES="default willneed willneed,random" bash $D/cold-b.sh $M/qwen3.6-35b-a3b-int4.giw
echo "finished $(date '+%H:%M:%S')"
