#!/usr/bin/env bash
# R-12, the Gemma 2 2B arm of the in-place ring window (pre-registered in ../r12-ring-direct-2026-10-05.md, committed before this runs).
# A pre-built test binary (built from 16e45a6d, so the tree may move before tonight), one process, the harness's own ABBA flips ringDirectDecode in-process.
# Nothing here needs a Claude session alive. Output: $D/g2-ab.log (durable) and the night runner's own log.
D=$HOME/goinfer-logs/r12
BIN=$D/decoder.test
MODEL=$HOME/models/gemma-2-2b-it-Q8_0.gguf
if [ ! -x "$BIN" ]; then echo "FATAL: $BIN is missing or not executable"; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "FATAL: $MODEL is missing"; exit 2; fi
case "$MODEL" in /srv/models/*) echo "FATAL: a model under /srv/models is not a bench surface"; exit 2;; esac
idle() { for i in $(seq 1 90); do l=$(cut -d" " -f1 /proc/loadavg); awk -v l="$l" "BEGIN{exit !(l<1.0)}" && return 0; echo "=== waiting for idle: loadavg $l"; sleep 20; done; return 1; }
idle || { echo "FATAL: box never went idle"; exit 3; }
cd $HOME/mycode/goinfer/decoder || exit 2
echo "=== $(date -u +%FT%TZ) START gemma2-2b depth 4500, 3 ABBA pairs"
GOINFER_HEAVY_TESTS=1 GOINFER_CPU_MODEL_G2=$MODEL "$BIN" -test.run 'TestCPURing_directDecodeAB/gemma2' -test.v -test.count=1 -test.timeout 60m > "$D/g2-ab.log" 2>&1
rc=$?
echo "=== $(date -u +%FT%TZ) END rc=$rc"
grep -E "pair|mean ON" "$D/g2-ab.log" | cut -c1-200
if ! grep -q "paired median" "$D/g2-ab.log"; then echo "FATAL: no result line in $D/g2-ab.log (the harness refused or died)"; exit 4; fi
exit $rc
