#!/usr/bin/env bash
# Q1 split, the card arms (README.md, "Q1 split"): C, goinfer's CUDA resident; then L, llama.cpp. Run by night.py, which holds
# the timing lock. Pre-built: the instrument binary in ~/goinfer-bench/gptoss-split-bin (sha256 e30c230b0f6d, built from
# split/namelogits_test.go.txt at c29890d7d); llama.cpp 427291b. Output in ~/goinfer-logs/mm01/gptoss-split.
set -u
B=~/goinfer-bench/gptoss-split-bin; S=~/goinfer-logs/mm01/gptoss-split; M=~/models/gpt-oss-20b-MXFP4.gguf
LL=~/mycode/peers/llama.cpp/build/bin
log() { echo "$(date '+%F %T') $*" | tee -a $S/card.log; }
void() {
  local apps load avail
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xwayland|plasmashell")
  load=$(cut -d' ' -f1 /proc/loadavg); avail=$(awk '/^MemAvailable:/{printf "%.1f", $2*1024/1e9}' /proc/meminfo)
  log "$1 start: load $load, MemAvailable $avail GB, driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader), card: ${apps:-only the compositor}"
  [ -n "$apps" ] && { log "VOID $1: card busy"; return 1; }
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 8.0 else 1)" "$load" || { log "VOID $1: load $load > 8.0"; return 1; }
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 15 else 1)" "$avail" || { log "VOID $1: MemAvailable $avail GB < 13 + 2"; return 1; }
}
rc=0
if void C; then
  NAMELOGITS_ARM=cuda NAMELOGITS_DIR=$S $B/namelogits.test -test.run '^TestScratchGptOssNameLogits$' -test.v -test.count=1 -test.timeout 20m > $S/c.log 2>&1
  c=$?; log "C exit $c"; grep -E 'loaded in|steps in|^(--- |ok|FAIL)' $S/c.log | tee -a $S/card.log; [ $c -eq 0 ] || rc=1
else rc=1; fi
if void L; then
  $LL/llama-server -m $M --host 127.0.0.1 --port 18294 -c 16384 -ngl 99 -ot '\.ffn_(up|down|gate)_exps\.=CPU' -np 1 -sp > $S/llama-server.log 2>&1 </dev/null & sp=$!
  t0=$(date +%s)
  until curl -sf http://127.0.0.1:18294/health | grep -q ok; do
    kill -0 $sp 2>/dev/null || { log "L: llama-server exited"; rc=1; break; }
    [ $(( $(date +%s) - t0 )) -gt 600 ] && { log "L: no health after 600 s"; kill $sp; rc=1; break; }
    sleep 2
  done
  python3 -c "import json; i=json.load(open('$S/prompt-ids.json')); f=json.load(open('$S/forced.json'))['forced']; print(json.dumps({'prompt':i+f,'n_predict':1,'temperature':0,'top_k':1,'cache_prompt':False,'n_probs':20}))" > $S/llama-req.json
  curl -s -m 900 http://127.0.0.1:18294/completion -H 'Content-Type: application/json' --data-binary @$S/llama-req.json > $S/llama-step22.json
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  python3 -c "import json; d=json.load(open('$S/llama-step22.json')); print('L step 22:', [(t['token'], round(t['logprob'],3)) for t in d['completion_probabilities'][0]['top_logprobs'][:8]])" 2>&1 | tee -a $S/card.log
else rc=1; fi
python3 $S/compare.py $S 2>&1 | tee -a $S/card.log
log "card arms done (exit $rc)"
exit $rc
