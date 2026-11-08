#!/usr/bin/env bash
# G-S10q-e by day on nobara (Francis, 2026-11-08), at main + the script fix (rev in ./rev), under the exclusive timing lock: first the
# fit check of the 30B at --moe-cache-experts by the server's own banner (a load that does not reach "serving on" stops the run), then the
# registered script unchanged in its rule (docs/measurements/multimodal-support-2026-10/s10q-qwen3vlmoe/run-gs10q-e-night.sh).
set -uo pipefail
B=$HOME/goinfer-bench/gs10q-e-604f7ddf
SRC=$HOME/goinfer-bench/gs10q-e-wt
OUT=$HOME/goinfer-logs/gs10q-e-2026-11-08
MODEL=$HOME/models/qwen3-vl-30b-a3b-instruct
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
mkdir -p $OUT
echo "== fit check start $(date '+%F %T %Z') load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"
$B/serve-cuda -backend cuda -require-backend -model "$MODEL" -quant int4 -moe-cache-experts -addr 127.0.0.1:18098 > $OUT/fit.log 2>&1 &
pid=$!
for _ in $(seq 1 1200); do grep -q 'serving on' $OUT/fit.log && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null
if ! grep -q 'serving on' $OUT/fit.log; then echo "STOP: the fit check did not reach a serving server (read $OUT/fit.log); the gate does not run"; tail -8 $OUT/fit.log; exit 1; fi
echo "fit: $(grep -iE 'fit:|KV plan:|expert|resident' $OUT/fit.log | head -5 | tr '\n' ' ' | cut -c1-600)"
echo "== gate start $(date '+%F %T %Z') load $(cut -d' ' -f1-3 /proc/loadavg)"
SRC=$SRC BIN=$B bash $SRC/docs/measurements/multimodal-support-2026-10/s10q-qwen3vlmoe/run-gs10q-e-night.sh $OUT
rc=$?
echo "== gate end $(date '+%F %T %Z') exit $rc"
exit $rc
