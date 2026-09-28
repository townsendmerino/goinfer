#!/bin/bash
# C0 graded runs (pre-registered in d805dd7c): qwen2.5-coder-1.5b (.gguf) then qwen2.5-7b (.int4.metal.giw), Metal, int4.
set -u
B=$HOME/goinfer-bench/confidence-c0-2026-09-27
cd $HOME/tmcode/goinfer/metal || exit 1
ts() { date '+%H:%M:%S'; }
gate() { w=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}'; then echo "$(ts) idle: load1=$l1"; return; fi; [ $w -ge 1200 ] && { echo "$(ts) NOT IDLE after 1200s (load1=$l1)"; exit 1; }; [ $((w % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 (${w}s)"; sleep 5; w=$((w+5)); done; }
run() { name=$1; model=$2; tok=$3
  case "$model" in /srv/models/*|/Volumes/*) echo "model on the archive"; exit 1;; esac
  gate; echo "$(ts) == $name start"
  GOINFER_C0_MODEL=$model GOINFER_C0_TOKENIZER=$tok GOINFER_C0_OUT=$B/c0-$name.jsonl \
    go test -tags goinfer_testhooks -run 'TestConfidenceC0$' -count=1 -v -timeout 40m . 2>&1 | grep -v "^=== RUN"
  echo "$(ts) == $name exit ${PIPESTATUS[0]}"; }
echo "$(ts) == C0 graded start; tree $(git rev-parse --short HEAD); load $(sysctl -n vm.loadavg)"
run 1.5b $HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf $HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
run 7b $HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw $HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
echo "$(ts) == DONE"
