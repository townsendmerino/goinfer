#!/usr/bin/env bash
# MEM10 tier 1 A/B (docs/measurements/mem10-2026-11/ab-registration.md), by day on nobara. The fit check by serve's banner first (the 26B at
# --moe-cache-experts -quant int4 -ctx 4352; a load that does not reach "serving on" stops the run), then the test binary directly under the
# exclusive timing lock.
set -uo pipefail
P=$HOME/goinfer-bench/mem10ab; WT=$HOME/goinfer-bench/mem10ab-wt; L=$P/logs
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
M=$HOME/models/gemma4-26b-q4_k_m.gguf
case "$(readlink -f $M)" in /srv/models*|/Volumes/*) echo "FATAL: archive"; exit 2;; esac
echo "== fit check $(date '+%F %T %Z') load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
$P/bin/serve-cuda -backend cuda -require-backend -model $M -quant int4 -moe-cache-experts -ctx 4352 -addr 127.0.0.1:18097 > $L/fit.log 2>&1 &
pid=$!; for _ in $(seq 1 600); do grep -q 'serving on' $L/fit.log && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null
grep -q 'serving on' $L/fit.log || { echo "STOP: the fit check did not reach a serving server (read $L/fit.log)"; tail -8 $L/fit.log; exit 1; }
echo "fit: $(grep -iE 'decode path|fit:|expert' $L/fit.log | head -4 | tr '\n' ' ' | cut -c1-500)"
echo "== A/B start $(date '+%F %T %Z')"
(cd $WT/backend/cuda && GOINFER_HEAVY_TESTS=1 GOINFER_MEM10_PAGEABLE_DIR=$HOME/goinfer-bench/mem10-pageable GOINFER_MEM10_AB_OUT=$L/ab.json \
  python3 $WT/scripts/timing_lock.py run --label mem10-ab -- $P/bin/cuda.test -test.run '^TestMEM10_pageableExpertSourceAB$' -test.v -test.count=1 -test.timeout 90m) > $L/ab.log 2>&1
echo "== A/B end $(date '+%F %T %Z') exit $?"
grep -E '^\[mem10ab|--- (PASS|FAIL)' $L/ab.log | cut -c1-400
