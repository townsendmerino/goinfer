#!/bin/bash
# Cold start to a finished reply, by read-ahead mode (exploratory): the bundle is evicted, the server is started on
# the CPU, and one greedy 24-token reply is taken, then a second. Reported: seconds from process start to serving, to
# the first reply and to the second, and how much of the file is in the page cache after each.
set -uo pipefail
D=$HOME/goinfer-bench/coldload; C=$HOME/goinfer-bench/selfcheck/cachestat.py
f=$1
evict() { python3 -c 'import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' "$1"; }
res() { python3 "$C" "$1" | sed 's/.*: //; s/ of .*(/ (/'; }
now() { date +%s.%N; }
ask() { curl -s -m 900 127.0.0.1:$1/v1/chat/completions -H 'content-type: application/json' -d '{"model":"'"$2"'","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"'"$3"'"}]}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"].replace("\n"," ")[:60])'; }
echo "part B on $(basename "$f") started $(date '+%H:%M:%S')"
port=18600
for rep in 1 2; do
  for mode in ${MODES:-default random sequential}; do
    port=$((port+1)); log=$D/b-$mode-$rep.log
    evict "$f"; t0=$(now)
    if [ $mode = default ]; then "$D/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & else XP_MADV=$mode "$D/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & fi
    pid=$!
    until grep -q 'goinfer serving on' "$log"; do kill -0 $pid 2>/dev/null || { echo "serve exited: $(tail -1 "$log")"; break; }; sleep 0.2; done
    t1=$(now); r1=$(res "$f")
    name=$(curl -s -m 20 127.0.0.1:$port/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    a1=$(ask $port "$name" "List the first five prime numbers and say in one sentence why 1 is not among them."); t2=$(now); r2=$(res "$f")
    a2=$(ask $port "$name" "Name three rivers in Europe and the sea each one reaches."); t3=$(now); r3=$(res "$f")
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    python3 -c "import sys; t0,t1,t2,t3=map(float,sys.argv[1:5]); print(f'rep $rep  $mode: serving at {t1-t0:5.1f}s ($r1) | first reply done at {t2-t0:5.1f}s ($r2) | second reply took {t3-t2:5.1f}s ($r3)')" $t0 $t1 $t2 $t3
    echo "      replies: [$a1] [$a2]"
  done
done
echo "part B finished $(date '+%H:%M:%S')"
