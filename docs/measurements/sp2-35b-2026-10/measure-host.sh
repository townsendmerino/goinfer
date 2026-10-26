#!/usr/bin/env bash
# The 35B's host memory, plain serve against serve --drafter (one binary): VmHWM, RssAnon/RssFile after a warm and a timed greedy
# request and 60 s later, the reply compared. Usage: measure-host.sh <serve>
set -u
S=$1; MODEL=$HOME/models/qwen3.6-35b-a3b-q4_k_m.int4.e4h.cuda.v17.giw; DR=$HOME/models/qwen36-35b-dflash; PORT=8131; OUT=~/goinfer-logs/sp2-35b-2026-10
REQ='{"model":"bench","temperature":0,"max_tokens":256,"messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number, with a docstring and three doctests."}]}'
arm() {
	local name=$1; shift
	GOINFER_MOE_CACHE_EXPERTS=1 "$S" -model bench=$MODEL -backend cuda -addr 127.0.0.1:$PORT -ctx 4608 -thinking off "$@" > $OUT/$name.log 2>&1 &
	local pid=$!
	until curl -s -o /dev/null 127.0.0.1:$PORT/v1/models; do sleep 1; kill -0 $pid 2>/dev/null || { echo "$name: serve exited"; tail -5 $OUT/$name.log; return; }; done
	local listen=$(awk '/VmRSS/{print $2}' /proc/$pid/status)
	curl -s 127.0.0.1:$PORT/v1/chat/completions -d "$REQ" -o /dev/null
	local t=$(curl -s -w '%{time_total}' 127.0.0.1:$PORT/v1/chat/completions -d "$REQ" -o $OUT/$name.reply.json)
	sleep 2
	local st=$(awk '/VmHWM|RssAnon|RssFile/{printf "%s %s ", $1, $2}' /proc/$pid/status)
	sleep 60
	local st60=$(awk '/VmRSS|RssAnon|RssFile/{printf "%s %s ", $1, $2}' /proc/$pid/status)
	echo "$name: VmRSS listening $listen kB; after requests [$st]; 60 s later [$st60]; second request ${t} s"
	grep -E 'fit:|C′ cache|speculative recurrent|block drafter attached' $OUT/$name.log | head -4 | sed 's/^/    /'
	kill $pid; wait $pid 2>/dev/null; sleep 3
}
echo "# $(date -Is) load $(cut -d' ' -f1 /proc/loadavg) avail $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
arm plain35
arm drafter35 -drafter $DR -drafter-width 4
python3 -c "import json,sys; a,b=(json.load(open(f))['choices'][0]['message']['content'] for f in sys.argv[1:]); print('drafter35 reply', 'IDENTICAL to plain' if a==b else 'DIFFERS from plain')" $OUT/plain35.reply.json $OUT/drafter35.reply.json
