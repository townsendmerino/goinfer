#!/usr/bin/env bash
# Q1's card step: llama.cpp's greedy decode of run F's exact prompt ids (identical to llama.cpp's own tokenization, step 1).
set -u
T=$CLAUDE_JOB_DIR/tmp; R=$T/ref; L=~/goinfer-logs/mm01/gptoss-ref; W=~/goinfer-bench/nobara-mm01-ref-wt
LL=~/mycode/peers/llama.cpp/build/bin; M=~/models/gpt-oss-20b-MXFP4.gguf
F=~/goinfer-bench/nobara-mm01-gptoss-wt/docs/measurements/gptoss-forced-call-2026-10-30/raw/f-main/repro.jsonl
log() { echo "$(date '+%F %T') $*" | tee -a $L/window.log; }
cd $W
log "window2: card released at 19:46 (coordinator); no card step starts after 21:05"
# the card was released early (coordinator, 19:46): no wait
python3 scripts/timing_lock.py run --wait 3000 --label 'nobara-mm01 gpt-oss ref: llama.cpp greedy (~6 min)' -- bash -c "
  L=$L; log() { echo \"\$(date '+%F %T') \$*\" | tee -a \$L/window.log; }
  [ \"\$(date +%H%M)\" -gt 2105 ] && { log 'STOP: past 21:05, no card step starts'; exit 9; }
  apps=\$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E 'kwin_wayland|Xwayland|plasmashell')
  [ -n \"\$apps\" ] && { log \"VOID: card busy: \$apps\"; exit 6; }
  log \"card clear; load \$(cut -d' ' -f1 /proc/loadavg); MemAvailable \$(awk '/^MemAvailable:/{printf \"%.1f\", \$2*1024/1e9}' /proc/meminfo) GB\"
  $LL/llama-server -m $M --host 127.0.0.1 --port 18294 -c 16384 -ngl 99 -ot '\.ffn_(up|down|gate)_exps\.=CPU' -np 1 -sp > \$L/llama-server.log 2>&1 </dev/null & sp=\$!
  t0=\$(date +%s); until curl -sf http://127.0.0.1:18294/health | grep -q ok; do kill -0 \$sp || { log 'llama-server exited (see llama-server.log)'; exit 3; }; [ \$((\$(date +%s)-t0)) -gt 600 ] && { kill \$sp; exit 4; }; sleep 2; done
  log \"llama-server up after \$((\$(date +%s)-t0)) s\"
  python3 -c \"import json; print(json.dumps({'prompt':json.load(open('$R/goinfer-ids.json')),'n_predict':400,'temperature':0,'top_k':1,'cache_prompt':False,'return_tokens':True}))\" > \$L/llama-req.json
  curl -s -m 1500 http://127.0.0.1:18294/completion -H 'Content-Type: application/json' --data-binary @\$L/llama-req.json > \$L/llama-completion.json
  kill \$sp; wait \$sp
  log 'llama.cpp done'
"
python3 $R/compare2.py $L $F 2>&1 | tee -a $L/window.log
log "end"
