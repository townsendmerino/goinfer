#!/bin/bash
# G-S12.4b-3e (docs/tasks/task-multimodal-support-2026-10.md, "S12.4b-3"): the served cells for the Gemma shape in the
# batched prefill under --backend webgpu. new = serve built at S12.4b-3's head, old = serve at S12.4b-2 (4d5cf269).
#   1. a Gemma 3 4B text prompt, two lengths, five alternating rounds (served_text_ttft.py --words);
#   2. a Gemma 4 E2B image turn, four rotated rounds of a warm-up and two timed requests an arm (vision_ttft.py).
# Runs with nobody watching: binaries, clients, the prompt file and the image are pinned copies in $B; night.py holds
# the timing lock. TOWER (default "-vision-device gpu") is the E2B cells' tower flag, the same on both arms.
set -u
B=${B:-$HOME/goinfer-bench/s12-4b-2026-10-12}
G3=${G3:-$HOME/models/gemma-3-4b-it}
E2B=${E2B:-$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf}
E2BV=${E2BV:-$HOME/models/gemma-4-E2B-unq}
TOWER=${TOWER--vision-device gpu}
OUT=${OUT:-$B/out-3e}
NEW=$B/bin/serve-webgpu-s3
OLD=$B/bin/serve-webgpu-new
mkdir -p "$OUT" || exit 2
for f in "$NEW" "$OLD"; do [ -x "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
for f in "$B/served_text_ttft.py" "$B/vision_ttft.py" "$B/gemma3_preprocess_image.png" "$G3" "$E2B" "$E2BV"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"
  echo "new: $(cat "$NEW.rev" 2>/dev/null)"; echo "old: $(cat "$OLD.rev" 2>/dev/null)"
  echo "gemma 3: $G3"; echo "e2b: $E2B (vision $E2BV, tower flag '$TOWER')"; echo "load: $(uptime)"
} > "$OUT/provenance.txt"
rc=0
python3 -u "$B/served_text_ttft.py" "$OUT/g3-text.json" --new "$NEW" --old "$OLD" --model "$G3" \
  --words "${WORDS:-400,1600}" --rounds "${ROUNDS:-5}" --log-dir "$OUT/serve-logs-g3" 2>&1 | tee "$OUT/g3-text.log"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
python3 - "$OUT/e2b-plan.json" "$NEW" "$OLD" "$E2B" "$E2BV" "$B/gemma3_preprocess_image.png" "${IMG_ROUNDS:-4}" "$TOWER" <<'PY'
import json, sys
out, new, old, model, vision, img, rounds, tower = sys.argv[1:]
def cell(name, serve, port=18654):
    return {"cell": name, "engine": "goinfer", "api": "openai", "port": port, "model": "", "media": "image", "base": img,
            "prompt": "Describe this image.",
            "cmd": [serve, "--model", model, "--vision", vision, *tower.split(), "--backend", "webgpu", "--addr", f"127.0.0.1:{port}"]}
cells = [cell("gemma-4-e2b-image-new", new), cell("gemma-4-e2b-image-old", old)]
for c in cells:
    for a in c["cmd"]:
        if a.startswith("/Volumes/") or a.startswith("/srv/models"):
            sys.exit(f"{a} is the archive")
json.dump({"rounds": int(rounds), "per_round": 2, "rotate": True, "cells": cells}, open(out, "w"), indent=1)
PY
python3 -u "$B/vision_ttft.py" "$OUT/e2b-plan.json" "$OUT/e2b-image" 2>&1 | tee "$OUT/e2b-image.log"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$OUT/provenance.txt"
exit "$rc"
