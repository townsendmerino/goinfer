#!/bin/bash
# Step 0 of docs/tasks/task-giw-cold-load-2026-10.md: the cold load on the Mac (exploratory, no bar; run by day on the
# owner's word). The Mellum MoE bundle and a dense 7B control built here. Estimate: 12 min.
set -uo pipefail
D=$HOME/goinfer-bench/coldload-mac; M=$HOME/models
echo "started $(date '+%Y-%m-%d %H:%M:%S %Z'); $(uptime | sed 's/.*load/load/'); $(sysctl -n hw.memsize | awk '{printf "%.0f GB RAM", $1/2^30}'); $(diskutil info / 2>/dev/null | awk -F: '/File System Personality/ {gsub(/^ +/,"",$2); print $2}')"
cat "$D/bin/BUILD.txt" | head -1
find "$D" -maxdepth 1 -name '7b.giw*' -delete
"$D/bin/prequant" -quant int4 -target cpu-arm64 -o "$D/7b.giw" "$M/qwen2.5-7b-instruct-q4_k_m.gguf" 2>&1 | tail -1
bash "$D/mac-a.sh" "$M/mellum2.int4.giw" "$D/7b.giw"
bash "$D/mac-c.sh" "$M/mellum2.int4.giw" 2
bash "$D/mac-c.sh" "$D/7b.giw" 2
find "$D" -maxdepth 1 -name '7b.giw*' -delete
echo "finished $(date '+%H:%M:%S')"
