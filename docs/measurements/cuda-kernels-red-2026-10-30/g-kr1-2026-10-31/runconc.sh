#!/bin/bash
# G-KR1 concurrent run: $1 = label, $2 = kernels -test.run regex. Package cuda's test binary runs beside it, as in the night job's go test ./...
L=$HOME/goinfer-bench/nobara-sb01-kr-logs
cd $HOME/goinfer-bench/nobara-sb01-kr-wt/cuda
$L/cuda.test -test.v -test.count=1 -test.timeout 4m > $L/$1.cuda-side.log 2>&1 &
CPID=$!
sleep 1
$L/run1.sh "$1" "$2"
kill $CPID 2>/dev/null; wait $CPID 2>/dev/null
echo "cuda side: $(grep -c '^--- PASS' $L/$1.cuda-side.log) passed, $(grep -c '^--- FAIL' $L/$1.cuda-side.log) failed before it was stopped"
