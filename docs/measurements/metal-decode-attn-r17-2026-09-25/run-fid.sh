#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
echo "$(date +%T) start commit $(git rev-parse --short HEAD) (+ uncommitted metal/decode_attn_r17_test.go)"
for M in qwen2.5-7b-instruct-q4_k_m.int4.metal.giw qwen2.5-coder-1.5b-instruct-q4_k_m.gguf; do
  echo "##### MODEL $M"; GOINFER_METAL_R17=1 GOINFER_METAL_R17_MODEL=$HOME/models/$M GOINFER_METAL_R17_DEPTHS=2048,3900 GOINFER_METAL_R17_SPLITS=0,16,32 GOINFER_METAL_R17_REPS=1 GOINFER_METAL_R17_TOKENS=1 \
    go test -tags goinfer_testhooks -count=1 -timeout 20m -v -run '^TestR17AttentionProto$' ./metal/ 2>&1; echo "##### exit=$?"
done
echo "$(date +%T) DONE"
