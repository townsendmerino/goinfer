#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
echo "$(date +%T) start commit $(git rev-parse --short HEAD) + uncommitted wiring; load1=$(sysctl -n vm.loadavg | awk '{print $2}')"
echo "##### SELECTION (heavy)"; GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 20m -v -run '^TestAttnFABlkSelection$' ./metal/ 2>&1; echo "##### exit=$?"
echo "##### R2 GATE set B through the production path $(date +%T)"; GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks -count=1 -timeout 60m -v -run '^TestR2_decodeFidelityGate$' ./metal/ 2>&1; echo "##### exit=$?"
echo "##### SUITE untagged $(date +%T)"; go test -count=1 -timeout 30m ./metal/ 2>&1 | tail -3; echo "##### exit=${PIPESTATUS[0]}"
echo "##### SUITE goinfer_testhooks $(date +%T)"; go test -count=1 -timeout 30m -tags goinfer_testhooks -v ./metal/ > /tmp/claude-501/wiring-suite-tagged.log 2>&1; echo "##### exit=$?"; grep -c -- "--- PASS" /tmp/claude-501/wiring-suite-tagged.log; grep -c -- "--- SKIP" /tmp/claude-501/wiring-suite-tagged.log; grep -E -- "--- FAIL" /tmp/claude-501/wiring-suite-tagged.log | head; tail -2 /tmp/claude-501/wiring-suite-tagged.log
echo "$(date +%T) DONE"
