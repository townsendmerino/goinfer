#!/usr/bin/env bash
# S1 on CUDA's speed record (docs/tasks/task-multimodal-support-2026-10.md, "S1 on CUDA"): Gemma 4 E2B decode, CUDA resident against CPU, as a night record
# (not a gate; no bar). Two parts, both timed, both under the timing lock:
#   1. bench_peer.py, goinfer only, E2B, Phase A at one depth, backends cpu and cuda from ONE serve binary (interleaved in one session by the harness);
#   2. the host's per-token PLE cost (TestGemma4EModel_realE2BPLEHostCost from a pre-built test binary).
# Estimate: ~25 min (two cells of 64 tok x 8 completions x 2 runs; the CPU cell dominates; plus loads and idle gates at BENCH_MAX_LOADAVG=1.0).
#
# The binaries are pre-built (BIN, from the commit in BIN/rev), so whatever the checkout holds tonight is not what runs:
#   CGO_ENABLED=0 go build -tags cuda -o $BIN/serve-cuda github.com/townsendmerino/goinfer/cuda/cmd/serve
#   (cd cuda && go test -c -tags 'cuda goinfer_testhooks' -o $BIN/cuda.test .)
#   python3 scripts/night.py add s1c-e2b-speed --est 30 --by "Claude, S1 on CUDA" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s1c-speed.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s1c}
OUT=${1:-$HOME/goinfer-bench/s1c/run-$(date +%F)}
MODEL=$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf
for p in "$BIN/serve-cuda" "$BIN/cuda.test" "$MODEL"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing"; exit 2; }
done
case "$MODEL" in /Volumes/*|/srv/models/*) echo "FATAL: $MODEL is on the archive (CLAUDE.md)"; exit 2;; esac
mkdir -p "$OUT"
{ echo "binaries: $BIN (rev $(cat "$BIN/rev" 2>/dev/null || echo unknown))"; echo "started:  $(date '+%F %T %Z')"
  echo "model:    $MODEL"; echo "gpu:      $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load:     $(cat /proc/loadavg)"; } | tee "$OUT/provenance.txt"
rc=0
echo "== 1/2 bench_peer E2B cpu+cuda: start $(date '+%T')"
GOINFER_SERVE_CPU="$BIN/serve-cuda" GOINFER_SERVE_CUDA="$BIN/serve-cuda" \
  BENCH_ENGINES=goinfer BENCH_BACKENDS=cpu,cuda BENCH_DEPTHS=none BENCH_MODELS=E2B \
  python3 "$SRC/scripts/bench_peer.py" "$OUT/bench-e2b.json" > "$OUT/bench-e2b.log" 2>&1 || { echo "!! bench_peer failed (see $OUT/bench-e2b.log)"; rc=1; }
tail -20 "$OUT/bench-e2b.log"
[ -s "$OUT/bench-e2b.json" ] || { echo "!! no results file: bench_peer exits 0 when it refuses a busy box"; rc=1; }
echo "== 2/2 PLE host cost: start $(date '+%T')"
cd "$SRC/cuda" || exit 2
python3 "$SRC/scripts/timing_lock.py" run --label s1c-ple-host -- \
  env GOINFER_HEAVY_TESTS=1 "$BIN/cuda.test" -test.run '^TestGemma4EModel_realE2BPLEHostCost$' -test.v -test.timeout 10m \
  > "$OUT/ple-host.log" 2>&1
grep -E "^(--- |PASS|FAIL)|ms/token" "$OUT/ple-host.log"
grep -q -- "--- PASS: TestGemma4EModel_realE2BPLEHostCost" "$OUT/ple-host.log" || { echo "!! the PLE host-cost test did not PASS (a skip or a failure)"; rc=1; }
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
