#!/bin/bash
# Cold start to a finished reply (exploratory), with the shipped loader (no experimental advice):
#   default  the load skips the CRC (the bundle carries its marker) and walks the headers
#   crc      GOINFER_GIW_VERIFY=always: the load reads the whole mapping front to back for the CRC first
#   cat      the file is read once with cat before the server starts (a sequential read through read(2))
set -uo pipefail
D=$HOME/goinfer-bench/coldload; C=$HOME/goinfer-bench/selfcheck/cachestat.py
f=$1; reps=${2:-2}
evict() { python3 -c 'import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)' "$1"; }
res() { python3 "$C" "$1" | sed 's/.*: //; s/ of .*(/ (/'; }
now() { date +%s.%N; }
ask() { curl -s -m 900 127.0.0.1:$1/v1/chat/completions -H 'content-type: application/json' -d '{"model":"'"$2"'","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"'"$3"'"}]}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"].replace("\n"," ")[:40])'; }
echo "part C on $(basename "$f") ($(stat -c %s "$f") bytes) started $(date '+%H:%M:%S')"
port=18700
for rep in $(seq 1 $reps); do
  for mode in default crc cat; do
    port=$((port+1)); log=$D/c-$(basename "$f")-$mode-$rep.log
    evict "$f"; t0=$(now)
    case $mode in
      default) "$D/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & ;;
      crc) GOINFER_GIW_VERIFY=always "$D/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & ;;
      cat) cat "$f" > /dev/null; "$D/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & ;;
    esac
    pid=$!
    until grep -q 'goinfer serving on' "$log"; do kill -0 $pid 2>/dev/null || { echo "serve exited: $(tail -1 "$log")"; break; }; sleep 0.2; done
    t1=$(now); r1=$(res "$f")
    name=$(curl -s -m 20 127.0.0.1:$port/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    a1=$(ask $port "$name" "List the first five prime numbers and say in one sentence why 1 is not among them."); t2=$(now); r2=$(res "$f")
    a2=$(ask $port "$name" "Name three rivers in Europe and the sea each one reaches."); t3=$(now)
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    python3 -c "import sys; t0,t1,t2,t3=map(float,sys.argv[1:5]); print(f'rep $rep  $mode: serving at {t1-t0:5.1f}s ($r1) | first reply done at {t2-t0:5.1f}s ($r2) | second reply took {t3-t2:5.1f}s   [$a1]')" $t0 $t1 $t2 $t3
  done
done
echo "part C finished $(date '+%H:%M:%S')"
