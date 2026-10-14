#!/bin/bash
# Night job: the windowed-KV step-compaction speed guard (registered in ../prefix-reuse-windowed-kv-2026-10-14.md section 8).
# Pinned binaries, no prompts, durable output. night.py holds the timing lock for the whole job.
#   OUT=<dir> overrides the output directory (the empty-dir test); SMOKE=1 runs the short smoke.
set -eo pipefail
B=$HOME/goinfer-bench/kvwindow-ring-2026-10-14
OUT=${OUT:-$HOME/goinfer-logs/kvwindow-ring-2026-10-14/speed-guard-$(date +%Y%m%d-%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for f in serve-main serve-new binaries.sha256; do
  [ -f "$B/$f" ] || { echo "missing $B/$f" >&2; exit 2; }
done
(cd "$B" && sha256sum -c binaries.sha256) | tee "$OUT/binaries-check.txt"
{ date; uname -r; nvidia-smi --query-gpu=driver_version,name,memory.used,temperature.gpu --format=csv,noheader; uptime; } > "$OUT/provenance.txt"
python3 -I "$HERE/speed_guard.py" "$B/serve-main" "$B/serve-new" "$OUT" 2>&1 | tee "$OUT/run.log"
exit "${PIPESTATUS[0]}"
