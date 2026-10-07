#!/usr/bin/env bash
# S1.9 of docs/tasks/task-multimodal-support-2026-10.md: Gemma 4 E2B decode speed, Metal resident against CPU, as a
# night record (not a gate). Two parts, both timed, both under the timing lock:
#   1. bench_peer.py, goinfer only, E2B, Phase A at one depth, backends cpu and metal from ONE serve binary
#      (interleaved in one session by the harness);
#   2. the host's per-token PLE cost (TestGemma4EModel_realE2BPLEHostCost from a pre-built test binary).
# Estimate: ~25 min (two cells of 64 tok x 8 completions x 2 runs; CPU E2B decodes ~20 tok/s; plus loads and idle
# gates), queued at 40.
#
# The binaries are pre-built (BIN, from the commit named in BIN/rev), so whatever the checkout holds tonight is not
# what runs:
#   go build -o $BIN/serve-metal ./metal/cmd/serve
#   (cd metal && go test -c -tags goinfer_testhooks -o $BIN/metal.test .)
#   python3 scripts/night.py add s19-e2b-speed --est 40 --by "Claude, S1" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s19-speed.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s19}
OUT=${1:-$HOME/goinfer-bench/s19/run-$(date +%F)}
MODEL=$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf
for p in "$BIN/serve-metal" "$BIN/metal.test" "$MODEL"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing"; exit 2; }
done
case "$MODEL" in /Volumes/*|/srv/models/*) echo "FATAL: $MODEL is on the archive (CLAUDE.md)"; exit 2;; esac
mkdir -p "$OUT"
{ echo "binaries: $BIN (rev $(cat "$BIN/rev" 2>/dev/null || echo unknown))"; echo "started:  $(date '+%F %T %Z')"
  echo "model:    $MODEL"; sw_vers 2>/dev/null | tr '\n' ' '; echo; pmset -g batt | head -1; } | tee "$OUT/provenance.txt"
rc=0
echo "== 1/2 bench_peer E2B cpu+metal: start $(date '+%T')"
GOINFER_SERVE_CPU="$BIN/serve-metal" GOINFER_SERVE_METAL="$BIN/serve-metal" \
  BENCH_ENGINES=goinfer BENCH_BACKENDS=cpu,metal BENCH_DEPTHS=none BENCH_MODELS=E2B \
  python3 "$SRC/scripts/bench_peer.py" "$OUT/bench-e2b.json" > "$OUT/bench-e2b.log" 2>&1 || { echo "!! bench_peer failed (see $OUT/bench-e2b.log)"; rc=1; }
tail -20 "$OUT/bench-e2b.log"
echo "== 2/2 PLE host cost: start $(date '+%T')"
cd "$SRC/metal" || exit 2
python3 "$SRC/scripts/timing_lock.py" run --label s19-ple-host -- \
  env GOINFER_HEAVY_TESTS=1 "$BIN/metal.test" -test.run '^TestGemma4EModel_realE2BPLEHostCost$' -test.v -test.timeout 10m \
  > "$OUT/ple-host.log" 2>&1
grep -E "^(--- |PASS|FAIL)|ms/token" "$OUT/ple-host.log"
grep -q -- "--- PASS: TestGemma4EModel_realE2BPLEHostCost" "$OUT/ple-host.log" || { echo "!! the PLE host-cost test did not PASS (a skip or a failure)"; rc=1; }
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
