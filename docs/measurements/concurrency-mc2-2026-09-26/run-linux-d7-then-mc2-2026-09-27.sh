#!/bin/bash
# Owed from 2026-09-26 (stopped twice for the owner): set-A D7-K512/1024 prefill refs, then the clean re-run of the
# MC2 Linux 0.5B depth-128 cell (the recorded one overlapped a 1.2 GB rsync). Expect D7 ~55-75 min, silent ~35 min.
export PATH=/usr/local/go/bin:$PATH
cd ~/mycode/goinfer || exit 1
ts() { date "+%F %T %Z"; }
echo "$(ts) start $(git rev-parse --short HEAD); $(uptime)"
echo "== $(ts) D7 regen (set A, K=512,1024)"
GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=a GOINFER_CPU_REF_KS=512,1024 go test -tags goinfer_testhooks -count=1 -timeout 6h -v -run "TestPrefillGateReference/D7$" ./decoder/ 2>&1
echo "$(ts) D7 exit $?"
w=0; while awk "BEGIN{exit !($(cut -d" " -f1 /proc/loadavg) > 1.0)}" && [ $w -lt 600 ]; do sleep 10; w=$((w+10)); done
echo "== $(ts) MC2 throughput 0.5b depth 128 int4 (clean re-run); $(cat /proc/loadavg)"
GOINFER_MC2=1 GOINFER_MC2_MODEL=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf GOINFER_MC2_DEPTH=128 go test -count=1 -timeout 30m -run "^TestMC2_batchedDecodeThroughput$" -v ./decoder/ 2>&1 | grep -E "^\[mc2|FAIL|panic|rep |METRIC|serial x1|batched B=|J8 workers"
echo "$(ts) DONE"
