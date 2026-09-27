#!/bin/bash
export PATH=/usr/local/go/bin:$PATH
cd ~/mycode/goinfer || exit 1
ts() { date "+%F %T %Z"; }
echo "$(ts) start $(git rev-parse --short HEAD); $(uptime)"
echo "== $(ts) MC2 identity on amd64 (0.5B int4 and int8int8)"
GOINFER_MC2_MODEL=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf go test -count=1 -timeout 30m -run "^TestMC2_decodeMultiStepBitIdentical$" -v ./decoder/ 2>&1 | grep -E "^(--- |    --- |ok|FAIL)|batchdecode_mc2_test"
for M in 0.5b 1.5b; do for D in 128 512; do
  echo "== $(ts) MC2 throughput $M depth $D int4; $(cat /proc/loadavg)"
  GOINFER_MC2=1 GOINFER_MC2_MODEL=$HOME/models/qwen2.5-coder-$M-instruct-q4_k_m.gguf GOINFER_MC2_DEPTH=$D go test -count=1 -timeout 30m -run "^TestMC2_batchedDecodeThroughput$" -v ./decoder/ 2>&1 | grep -E "^\[mc2|FAIL|panic"
done; done
echo "== $(ts) D7 regen (set A, K=512,1024)"
GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=a GOINFER_CPU_REF_KS=512,1024 go test -tags goinfer_testhooks -count=1 -timeout 6h -v -run "TestPrefillGateReference/D7$" ./decoder/ 2>&1
echo "$(ts) D7 exit $?"
echo "$(ts) DONE"
