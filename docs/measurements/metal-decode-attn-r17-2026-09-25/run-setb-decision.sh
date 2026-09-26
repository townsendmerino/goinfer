#!/bin/bash
# The set-B decision run pre-registered in docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md.
cd /Users/francistownsend-merino/tmcode/goinfer
echo "$(date +%T) start commit $(git rev-parse --short HEAD); tree: $(git status --short | tr '\n' ' ')"
export GOINFER_PREFILL_GATE_PROMPTS=b
for M in qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.giw qwen2.5-7b-instruct-q4_k_m.int4.metal.giw; do
  echo "##### P1 $M $(date +%T) load1=$(sysctl -n vm.loadavg | awk '{print $2}')"
  GOINFER_METAL_R17=1 GOINFER_METAL_R17_MODEL=$HOME/models/$M GOINFER_METAL_R17_PROMPTS=10 GOINFER_METAL_R17_DEPTHS=2048,3900 \
    GOINFER_METAL_R17_ACC_ARMS=decision go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR17KernelAccuracy$' ./metal/ 2>&1
  echo "##### exit=$?"
done
for spec in attention_fa:0 proto:16 exact-null:0; do
  c=${spec%%:*}; s=${spec##*:}
  echo "##### P2 cand=$c S=$s $(date +%T) load1=$(sysctl -n vm.loadavg | awk '{print $2}')"
  GOINFER_HEAVY_TESTS=1 GOINFER_METAL_R17_CAND=$c GOINFER_METAL_R17_SPLIT=$s go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR17_decodeFidelityGate$' ./metal/ 2>&1
  echo "##### exit=$?"
done
echo "$(date +%T) DONE"
