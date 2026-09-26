#!/bin/bash
# R17 precondition 4 (+3), parameters fixed in docs/measurements/metal-decode-attn-fidelity-setb-PREREGISTERED.md.
cd /Users/francistownsend-merino/tmcode/goinfer
wait_idle() { while :; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); if awk -v l=$l 'BEGIN{exit !(l<2.0)}'; then echo "$(date +%T) idle: load1=$l"; return; fi; echo "$(date +%T) waiting for idle, load1=$l"; sleep 30; done; }
echo "$(date +%T) start commit $(git rev-parse --short HEAD); tree: $(git status --short | tr '\n' ' ')"
wait_idle
GOINFER_METAL_R17=1 GOINFER_METAL_R17_DEPTHS=3900 GOINFER_METAL_R17_SPLITS=16 GOINFER_METAL_R17_REPS=7 GOINFER_METAL_R17_TOKENS=20 \
  go test -tags goinfer_testhooks -count=1 -timeout 30m -v -run '^TestR17AttentionProto$' ./metal/ 2>&1
echo "##### exit=$?"
echo "$(date +%T) DONE"
