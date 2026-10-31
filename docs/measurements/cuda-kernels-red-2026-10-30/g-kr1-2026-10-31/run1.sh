#!/bin/bash
# G-KR1 run: $1 = label, $2 = -test.run regex
L=$HOME/goinfer-bench/nobara-sb01-kr-logs
cd $HOME/goinfer-bench/nobara-sb01-kr-wt/cuda/kernels
{
echo "=== $1  $(date '+%F %T %Z')  load $(cut -d' ' -f1 /proc/loadavg)"
nvidia-smi --query-gpu=memory.used,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "run: $2"
timeout 300 $L/kernels.test -test.v -test.count=1 -test.timeout 4m -test.run "$2" 2>&1
echo "exit $?"
} > $L/$1.log 2>&1
grep -E "^(--- |=== RUN|ok|FAIL|PASS|exit)|ILLEGAL|MiB/alloc" $L/$1.log | grep -v "=== RUN" | head -40
