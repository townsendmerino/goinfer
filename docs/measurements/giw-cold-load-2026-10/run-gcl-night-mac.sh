#!/bin/bash
# The Mac's night run for docs/completed/task-giw-cold-load-2026-10.md, "The darwin pre-read": G-CLd2 (the cold start on
# the CPU and on Metal) and G-CLd3 (the warm starts), with old and new Metal-module servers as pinned binaries
# (bin/BUILD.txt names the revisions), plus an old-against-old pair on one CPU and one Metal cell for the noise.
# The dense and Metal bundles are built here, in the scratch directory, at the weights format both servers read.
# Estimate: 15 min.
set -uo pipefail
B=$HOME/goinfer-bench/coldload-mac-gate
M=$HOME/models
D=$HOME/goinfer-logs/giw-cold-load/$(date +%Y-%m-%d)
SMOKE=${GCL_SMOKE:-}   # GCL_SMOKE=1: one pair of two cells, exploratory, to check the tooling
mkdir -p "$D"
for f in "$B/bin/serve-old" "$B/bin/serve-new" "$B/bin/prequant" "$B/gcl_night.py" "$M/mellum2.int4.giw" "$M/qwen2.5-7b-instruct-q4_k_m.gguf" "$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"; do
  [ -e "$f" ] || { echo "missing: $f"; exit 2; }
done
build() { # build <name> <gguf> <target>
  find "$B" -maxdepth 1 -name "$1*" -delete
  "$B/bin/prequant" -quant int4 -target "$3" -o "$B/$1" "$M/$2" 2>&1 | tail -1
  [ -s "$B/$1" ] || { echo "$1 did not build"; exit 2; }
}
L=$M/mellum2.int4.giw
if [ -n "$SMOKE" ]; then
  build 15b.metal.giw qwen2.5-coder-1.5b-instruct-q4_k_m.gguf metal
  python3 "$B/gcl_night.py" --bin-dir "$B/bin" --log-dir "$D/smoke-logs" --record "$D/gcl-mac-smoke.jsonl" --est-min 2 \
    "smoke-cold-mellum-cpu|$L|cold|1|old:new||0|cpu" \
    "smoke-cold-1.5b-metal|$B/15b.metal.giw|cold|1|old:new||0|metal" 2>&1 | tee "$D/gcl-mac-smoke.log"
  rc=${PIPESTATUS[0]}
  find "$B" -maxdepth 1 -name '*.giw*' -delete
  exit $rc
fi
build 7b.cpu.giw qwen2.5-7b-instruct-q4_k_m.gguf cpu-arm64
build 7b.metal.giw qwen2.5-7b-instruct-q4_k_m.gguf metal
build 15b.metal.giw qwen2.5-coder-1.5b-instruct-q4_k_m.gguf metal
python3 "$B/gcl_night.py" --bin-dir "$B/bin" --log-dir "$D/logs" --record "$D/gcl-mac.jsonl" --est-min 15 \
  "aa-cold-mellum-cpu|$L|cold|2|old:old||0|cpu" \
  "cold-mellum-cpu|$L|cold|4|old:new||0|cpu" \
  "cold-7b-cpu|$B/7b.cpu.giw|cold|4|old:new||0|cpu" \
  "aa-cold-7b-metal|$B/7b.metal.giw|cold|2|old:old||0|metal" \
  "cold-7b-metal|$B/7b.metal.giw|cold|4|old:new||0|metal" \
  "cold-1.5b-metal|$B/15b.metal.giw|cold|4|old:new||0|metal" \
  "warm-mellum-cpu|$L|warm|4|old:new||0|cpu" \
  "warm-7b-cpu|$B/7b.cpu.giw|warm|4|old:new||0|cpu" \
  "warm-7b-metal|$B/7b.metal.giw|warm|4|old:new||0|metal" \
  "warm-1.5b-metal|$B/15b.metal.giw|warm|4|old:new||0|metal" \
  2>&1 | tee "$D/gcl-mac.log"
rc=${PIPESTATUS[0]}
find "$B" -maxdepth 1 -name '*.giw*' -delete
# A cold cell in which the new build did not prefetch every time, or a Metal cell that did not run on Metal, is not
# run, and a cell that is not run is not a pass.
python3 - "$D/gcl-mac.jsonl" <<'PY' || rc=1
import json, sys
bad = 0
for line in open(sys.argv[1]):
    d = json.loads(line)
    if d.get("event") != "summary" or not d.get("pairs"):
        continue
    if d["cell"].startswith("cold-") and d.get("b_read_first_announced") != d["pairs"]:
        print(f"NOT RUN: {d['cell']}: the new build prefetched in {d.get('b_read_first_announced')} of {d['pairs']} starts"); bad = 1
    if d["cell"].endswith("-metal") and not all("metal" in p for p in d.get("decode_paths", [])):
        print(f"NOT RUN: {d['cell']}: not every start ran on Metal: {d.get('decode_paths')}"); bad = 1
sys.exit(bad)
PY
exit $rc
