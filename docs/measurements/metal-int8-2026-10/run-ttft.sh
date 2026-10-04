#!/usr/bin/env bash
# Night job: S-auto's first-token half (docs/tasks/task-metal-int8-2026-10.md, "S-auto's first-token half"),
# pre-registered 2026-10-04 before any graded run: TestW8Native_S_ttft on the 0.5B and the 1.5B, a 1,000-token prompt,
# Metal native int8 (precise math) against CPU int8int8, Metal int4 re-quant reported, 7 rotated repetitions; warm is
# graded, cold reported. The test builds its own prompt, so the binary needs no package directory.
#   go test -c -tags goinfer_testhooks -o ~/goinfer-bench/metal-int8-2026-10/metal-<REV>.test ./metal/
# Queued with:
#   python3 scripts/night.py add metal-int8-ttft --est 10 --by "Claude, Metal int8 S-auto TTFT" \
#     --doc docs/tasks/task-metal-int8-2026-10.md -- bash docs/measurements/metal-int8-2026-10/run-ttft.sh
# The runner holds the timing lock. Logs: ~/goinfer-logs/metal-int8-2026-10/ttft/.
set -uo pipefail
REV=d0d65266
BASE=$HOME/goinfer-bench/metal-int8-2026-10
BIN=$BASE/metal-$REV.test
LOG=$HOME/goinfer-logs/metal-int8-2026-10/ttft
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"
env GOINFER_HEAVY_TESTS=1 GOINFER_W8_GATE_TTFT=1 "$BIN" -test.v -test.count=1 -test.timeout 40m \
  -test.run '^TestW8Native_S_ttft$' 2>&1 | tee "$LOG/ttft.log"
s=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), exit $s" | tee -a "$LOG/provenance.txt"
exit "$s"
