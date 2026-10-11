#!/bin/bash
# nobara's night run for docs/completed/task-giw-cold-load-2026-10.md: the timed gates G-CL2 (the MoE cold start), G-CL3
# (the dense cold start and every warm start) and G-CL4 (the bundle that does not fit: the 35B paged, in a 16 GB
# memory scope), each with old and new CPU servers as pinned binaries (bin/BUILD.txt names the revisions), plus an
# old-against-old pair of cold starts on each MoE bundle for the noise. Estimate: 45 min.
set -uo pipefail
B=$HOME/goinfer-bench/coldload-gate
M=$HOME/models
D=$HOME/goinfer-logs/giw-cold-load/$(date +%Y-%m-%d)
mkdir -p "$D"
for f in "$B/bin/serve-old" "$B/bin/serve-new" "$B/bin/prequant" "$B/gcl_night.py" "$M/qwen3.6-35b-a3b-int4.giw" "$M/mellum2.int4.giw" "$M/qwen2.5-7b-instruct-q4_k_m.gguf"; do
  [ -e "$f" ] || { echo "missing: $f"; exit 2; }
done
# The dense control, at the weights format both servers read.
find "$B" -maxdepth 1 -name '7b.giw*' -delete
"$B/bin/prequant" -quant int4 -target cpu-amd64 -o "$B/7b.giw" "$M/qwen2.5-7b-instruct-q4_k_m.gguf" 2>&1 | tail -1
[ -s "$B/7b.giw" ] || { echo "the dense control did not build"; exit 2; }
Q=$M/qwen3.6-35b-a3b-int4.giw; L=$M/mellum2.int4.giw; S=$B/7b.giw
python3 "$B/gcl_night.py" --bin-dir "$B/bin" --log-dir "$D/logs" --record "$D/gcl-nobara.jsonl" --est-min 45 \
  "aa-cold-35b|$Q|cold|2|old:old||0" \
  "cold-35b|$Q|cold|4|old:new||0" \
  "aa-cold-mellum|$L|cold|2|old:old||0" \
  "cold-mellum|$L|cold|4|old:new||0" \
  "cold-7b|$S|cold|4|old:new||0" \
  "warm-35b|$Q|warm|4|old:new||0" \
  "warm-mellum|$L|warm|4|old:new||0" \
  "warm-7b|$S|warm|4|old:new||0" \
  "nofit-35b|$Q|cold|4|old:new|-stream-weights -weight-cache 8 -accept-slow|16" \
  2>&1 | tee "$D/gcl-nobara.log"
rc=${PIPESTATUS[0]}
find "$B" -maxdepth 1 -name '7b.giw*' -delete
exit $rc
