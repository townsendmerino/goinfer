#!/bin/bash
# KS-H2 readings, second attempt: the first (readings-1-h2void-h3ok.log) ran with another process's 6.9 GB CUDA context still on the card.
# Each reading waits up to 5 minutes for a card with nothing but the compositor on it; otherwise it is VOID and does not run.
set -u
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
cd $HOME/goinfer-bench/ks-h2-h3
echo "tree $(git rev-parse --short HEAD); driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); start $(date '+%F %T %Z'); load $(cut -d' ' -f1-3 /proc/loadavg)"
clear_card() {
  for i in $(seq 60); do
    others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -v kwin_wayland)
    [ -z "$others" ] && { echo "== card before $1 (only the compositor):"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; return 0; }
    sleep 5
  done
  echo "VOID $1: another CUDA context on the card after 5 minutes: $others"; return 1
}
if clear_card H2-R1; then
  KSH2_MODEL=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf go test -count=1 -tags 'cuda goinfer_testhooks' ./cuda/ -run '^TestKSH2_buildVRAM$' -v -timeout 15m 2>&1
fi
if clear_card H2-R2; then
  KSH2_MODEL=$HOME/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf KSH2_CTX=16384 go test -count=1 -tags 'cuda goinfer_testhooks' ./cuda/ -run '^TestKSH2_buildVRAM$' -v -timeout 15m 2>&1
fi
echo "end of H2 $(date '+%F %T %Z')"
# KS-O1-F D1 then D3 (registered at 6452689d2), in the same lock hold so they do not wait out the night queue's next job.
bash /home/francis/goinfer-logs/ks-o1-f-2026-10-25/run-d1.sh > /home/francis/goinfer-logs/ks-o1-f-2026-10-25/d1.log 2>&1
KSO1F_KV=f32 bash /home/francis/goinfer-logs/ks-o1-f-2026-10-25/run-d1.sh > /home/francis/goinfer-logs/ks-o1-f-2026-10-25/d3.log 2>&1
echo "end $(date '+%F %T %Z')"
