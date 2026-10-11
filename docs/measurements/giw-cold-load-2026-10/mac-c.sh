#!/bin/bash
# Step 0 on the Mac (exploratory, no bar), part C: cold start to a finished reply on the CPU.
#   default  the load as shipped on darwin (no advice, no CRC: the bundle carries its marker)
#   random   the read-around off for the header walk, the default after it (XP_MADV=random)
#   crc      GOINFER_GIW_VERIFY=always: the whole mapping is read front to back for the CRC before the walk
#   cat      the file is read once with cat before the server starts
set -uo pipefail
D=$HOME/goinfer-bench/coldload-mac
f=$1; reps=${2:-2}
res() { python3 "$D/maccache.py" resident "$1" | sed 's/.*: //; s/ of .*//'; }
now() { python3 -c 'import time; print(time.time())'; }
ask() { curl -s -m 900 127.0.0.1:$1/v1/chat/completions -H 'content-type: application/json' -d '{"model":"'"$2"'","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"'"$3"'"}]}' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"].replace("\n"," ")[:40])'; }
echo "part C on $(basename "$f") ($(stat -f %z "$f") bytes) started $(date '+%H:%M:%S')"
port=18950
for rep in $(seq 1 $reps); do
  for mode in ${MODES:-default random crc cat}; do
    port=$((port+1)); log=$D/c-$(basename "$f")-$mode-$rep.log
    python3 "$D/maccache.py" invalidate "$f" > /dev/null; r0=$(res "$f"); t0=$(now)
    case $mode in
      default) "$D/bin/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & ;;
      random) XP_MADV=random "$D/bin/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & ;;
      crc) GOINFER_GIW_VERIFY=always "$D/bin/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & ;;
      cat) cat "$f" > /dev/null; "$D/bin/serve-xp" -model "$f" -backend cpu -addr 127.0.0.1:$port > "$log" 2>&1 & ;;
    esac
    pid=$!
    until grep -q 'goinfer serving on' "$log"; do kill -0 $pid 2>/dev/null || { echo "serve exited: $(tail -2 "$log" | cut -c1-200)"; break; }; sleep 0.2; done
    t1=$(now); r1=$(res "$f")
    name=$(curl -s -m 20 127.0.0.1:$port/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    a1=$(ask $port "$name" "List the first five prime numbers and say in one sentence why 1 is not among them."); t2=$(now); r2=$(res "$f")
    a2=$(ask $port "$name" "Name three rivers in Europe and the sea each one reaches."); t3=$(now)
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    python3 -c "import sys; t0,t1,t2,t3=map(float,sys.argv[1:5]); print(f'rep $rep  $mode: resident before $r0 | serving at {t1-t0:5.1f}s ($r1) | first reply done at {t2-t0:5.1f}s ($r2) | second reply took {t3-t2:5.1f}s   [$a1]')" $t0 $t1 $t2 $t3
  done
done
echo "part C finished $(date '+%H:%M:%S')"
