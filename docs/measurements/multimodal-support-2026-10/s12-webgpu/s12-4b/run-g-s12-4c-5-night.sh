#!/bin/bash
# G-S12.4c-5 (docs/tasks/task-multimodal-support-2026-10.md, "S12.4c"): the served re-read of G-S12.4b-3e's Gemma 3 4B
# text cells under --backend webgpu on nobara, the build with the wide key-split attention kernel and its rows form
# against the build before them. Five alternating rounds, two prompt lengths (served_text_ttft.py --words).
# Runs with nobody watching: both serve binaries and the client are pinned copies in $B; night.py holds the timing lock.
#
#   B=~/goinfer-bench/s12-4b-2026-10-12 bash $B/run-g-s12-4c-5-night.sh
set -u
B=${B:-$HOME/goinfer-bench/s12-4b-2026-10-12}
G3=${G3:-$HOME/models/gemma-3-4b-it}
OUT=${OUT:-$B/out-4c-5}
NEW=$B/bin/serve-webgpu-4c
OLD=$B/bin/serve-webgpu-s3
mkdir -p "$OUT" || exit 2
for f in "$NEW" "$OLD"; do [ -x "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
for f in "$B/served_text_ttft.py" "$G3"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
case "$G3" in /Volumes/*|/srv/models*) echo "FATAL: $G3 is the archive"; exit 2;; esac
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"
  echo "new: $(cat "$NEW.rev" 2>/dev/null)"; echo "old: $(cat "$OLD.rev" 2>/dev/null)"
  echo "gemma 3: $G3"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; echo "load: $(uptime)"
} > "$OUT/provenance.txt"
python3 -u "$B/served_text_ttft.py" "$OUT/g3-text.json" --new "$NEW" --old "$OLD" --model "$G3" \
  --words "${WORDS:-400,1600}" --rounds "${ROUNDS:-5}" --log-dir "$OUT/serve-logs" 2>&1 | tee "$OUT/g3-text.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$OUT/provenance.txt"
exit "$rc"
