#!/bin/bash
# KS-O1-F (c), the filler generality read (registered at a030b4cfa). A night.py job (the exclusive lock): both Phi-3 arms need nearly the whole card, so it
# waits up to 5 minutes for a card holding only the compositor and otherwise fails. D2a's first attempt ran out of
# device memory beside another session's serve under the shared lock.
set -uo pipefail
export PATH=/usr/local/go/bin:$PATH
cd ~/goinfer-bench/ks-o1-f || exit 1
echo "== $(date -Is) commit $(git rev-parse --short HEAD)"
for i in $(seq 60); do
  others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -v kwin_wayland)
  [ -z "$others" ] && break
  sleep 5
done
[ -n "$others" ] && { echo "FAILED: another CUDA context on the card after 5 minutes: $others"; exit 1; }
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
uptime
GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestKSO1F_fillerGenerality$' -v ./cuda/ 2>&1 | tee /home/francis/goinfer-logs/ks-o1-f-2026-10-25/c.log
rc=${PIPESTATUS[0]}; echo "== exit $rc $(date -Is)"; exit $rc
