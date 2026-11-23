#!/bin/bash
# The fix's served A/B (README "Fix"), by day, under the exclusive timing lock:
#   OLD=<serve before the fix> NEW=<serve with it> OUT=<dir> python3 scripts/timing_lock.py run --label k56-ab -- bash run-fix-ab.sh
# Phi-3 mini at its default (q4k) on CUDA, depth 128, goinfer (NEW) against goinfer_old (OLD), BENCH_RUNS=3, BENCH_ABBA=1. No
# BENCH_REQUIRE_BACKEND: serve's --require-backend refuses a load that declines any path, and q4k declines the batched prefill. Before the
# timed cells, each binary is started once to read its fit banner and its device memory, which also makes NEW write its .k56 sidecar,
# so no timed cell builds one.
set -uo pipefail
: "${OLD:?}" "${NEW:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
MODEL=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
mkdir -p "$OUT/serve-logs" || exit 2
{ echo "started: $(date '+%F %T %Z')"; echo "old: $OLD sha256 $(sha256sum "$OLD" | cut -c1-16)"; echo "new: $NEW sha256 $(sha256sum "$NEW" | cut -c1-16)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"; } | tee "$OUT/provenance.txt"
banner() { # banner <name> <bin>: start the server, read its banner and its device memory once it listens, stop it
  local log=$OUT/banner-$1.log
  "$2" -model bench="$MODEL" -backend cuda -ctx 4096 -addr 127.0.0.1:18241 > "$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do grep -q 'goinfer serving on' "$log" && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
  local mib; mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
  kill $pid; wait $pid 2>/dev/null
  echo "$1: device ${mib} MiB | $(grep -m1 ' fit:' "$log" | sed 's/^ *//') | $(grep -m1 'swap guard (load): armed for' "$log" | sed 's/.*armed for //; s/:.*//')"
}
for _ in $(seq 1 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && break; sleep 10; done
banner new-first "$NEW" | tee -a "$OUT/banners.txt"   # writes the .k56 sidecar if it is not there
banner new "$NEW" | tee -a "$OUT/banners.txt"
banner old "$OLD" | tee -a "$OUT/banners.txt"
for _ in $(seq 1 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && break; sleep 10; done
(cd "$REPO" && env BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_MODELS=phi3-mini BENCH_DEPTHS=none BENCH_PHASES=A \
  BENCH_QUANT_OVERRIDE="phi3-mini=default" BENCH_RUNS=3 BENCH_ABBA=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs" \
  GOINFER_SERVE_CUDA="$NEW" GOINFER_SERVE_CUDA_OLD="$OLD" python3 -u scripts/bench_peer.py "$OUT/decode-cuda.json") 2>&1 | tee "$OUT/decode-cuda.log"
rc=${PIPESTATUS[0]}
# bench_peer exits 0 with errored cells (a server that never came up) and when it refuses a busy box: count the cells that ran.
python3 - "$OUT/decode-cuda.json" <<'PY' || rc=1
import json, sys
try:
    cells = [c for c in json.load(open(sys.argv[1])) if isinstance(c, dict) and "engine" in c]
except (OSError, ValueError) as e:
    sys.exit(f"!! no results: {e}")
bad = [f"{c['engine']} block {c.get('block')}: {c.get('error')}" for c in cells if c.get("error") or not c.get("runs")]
if not cells or bad:
    sys.exit(f"!! {len(cells)} cell(s), {len(bad)} errored or empty: {bad[:4]}")
print(f"{len(cells)} cells, none errored")
PY
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
