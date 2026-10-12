#!/bin/bash
# G-S12.4b-2d (docs/tasks/task-multimodal-support-2026-10.md, "S12.4b"): served TTFT of a long text prompt under
# --backend webgpu at the default quant, the batched int4 prefill against the build before it, two arms interleaved.
# Runs with nobody watching: both serve binaries are built beforehand and pinned in $B/bin (their revisions in
# <binary>.rev), the client and the prompts are copies in $B, and night.py holds the timing lock for the job.
#
#   B=~/goinfer-bench/s12-4b-2026-10-12 bash $B/run-g-s12-4b-2d-night.sh
set -u
B=${B:-$HOME/goinfer-bench/s12-4b-2026-10-12}
MODEL=${MODEL:-$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}
OUT=$B/out
mkdir -p "$OUT" || exit 2
for f in "$B/bin/serve-webgpu-new" "$B/bin/serve-webgpu-old"; do
  [ -x "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
for f in "$B/served_text_ttft.py" "$B/prompts.json" "$MODEL"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
{
  echo "started: $(date '+%F %T %Z')"
  echo "host: $(uname -srm)"
  echo "new: $(cat "$B/bin/serve-webgpu-new.rev" 2>/dev/null)"
  echo "old: $(cat "$B/bin/serve-webgpu-old.rev" 2>/dev/null)"
  echo "model: $MODEL"
  echo "load: $(uptime)"
} > "$OUT/provenance.txt"
python3 -u "$B/served_text_ttft.py" "$OUT/g-s12-4b-2d.json" \
  --new "$B/bin/serve-webgpu-new" --old "$B/bin/serve-webgpu-old" --model "$MODEL" \
  --prompts "$B/prompts.json" --key 1.5B --depths "${DEPTHS:-512,2048}" --rounds "${ROUNDS:-5}" \
  --log-dir "$OUT/serve-logs" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$OUT/provenance.txt"
exit "$rc"
