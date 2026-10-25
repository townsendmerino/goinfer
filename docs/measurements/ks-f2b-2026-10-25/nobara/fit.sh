#!/bin/bash
# KS-F2b on nobara: build both arms' test binaries, then the fit check (one B run, depth 128, 2 tokens, per model) under a memory cap
# with no swap, so a load that does not fit is killed instead of swapping the box.
set -u
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
L=$HOME/goinfer-logs/ks-f2b-nobara-2026-10-25; B=$L/bins
S=$HOME/goinfer-bench/ks-f2b-nobara/docs/measurements/ks-f2b-2026-10-25/run_ksf2b_linux.py
echo "== $(date '+%F %T %Z') build; load $(cut -d' ' -f1-3 /proc/loadavg)"
(cd $HOME/goinfer-bench/ks-f2b-A && go test -c -o $B/decoder-A.test ./decoder/) || { echo "FAILED: build A"; exit 1; }
(cd $HOME/goinfer-bench/ks-f2b-B && go test -c -o $B/decoder-B.test ./decoder/) || { echo "FAILED: build B"; exit 1; }
echo "A $(cd $HOME/goinfer-bench/ks-f2b-A && git rev-parse --short HEAD), B $(cd $HOME/goinfer-bench/ks-f2b-B && git rev-parse --short HEAD)"
for m in granite/granite-4.0-h-tiny-Q8_0.gguf nemotron-h-4b-instruct-128k qwen3.6-35b-a3b-q4_k_m.gguf llama4-scout-gguf/Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf; do
  echo "== $(date '+%T') fit $m"
  systemd-run --user --scope -q -p MemoryMax=52G -p MemorySwapMax=0 -- python3 $S $B $HOME/models/$m 128 0 $L/fit.jsonl --fit 2>&1 | tail -4
  echo "   rc ${PIPESTATUS[0]}"
done
echo "== end $(date '+%F %T %Z')"
