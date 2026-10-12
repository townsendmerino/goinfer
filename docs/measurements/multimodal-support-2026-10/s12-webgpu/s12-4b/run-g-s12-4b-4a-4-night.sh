#!/bin/bash
# G-S12.4b-4a-4 (docs/tasks/task-multimodal-support-2026-10.md, "S12.4b-4a"): Qwen2.5-VL 3B's S7 image cell under
# --backend webgpu on nobara (Vulkan), the build whose m-RoPE image prefill runs on the device against the build
# before it (the CPU's prefill and an upload). Rotated rounds of a warm-up and two timed requests an arm, a fresh
# image a timed request (vision_ttft.py). Ships at a median new/old time to first token of 0.98 or under.
# Runs with nobody watching: both serve binaries, the client and the image are pinned copies in $B; night.py holds
# the timing lock.
#
#   B=~/goinfer-bench/s12-4b-2026-10-12 bash $B/run-g-s12-4b-4a-4-night.sh
set -u
B=${B:-$HOME/goinfer-bench/s12-4b-2026-10-12}
MODEL=${MODEL:-$HOME/models/qwen25vl-3b-instruct}
OUT=${OUT:-$B/out-4a-4}
NEW=$B/bin/serve-webgpu-mrope
OLD=$B/bin/serve-webgpu-4c
IMG=$B/gemma3_preprocess_image.png
mkdir -p "$OUT" || exit 2
for f in "$NEW" "$OLD"; do [ -x "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
for f in "$B/vision_ttft.py" "$IMG" "$MODEL"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
case "$MODEL" in /Volumes/*|/srv/models*) echo "FATAL: $MODEL is the archive"; exit 2;; esac
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"
  echo "new: $(cat "$NEW.rev" 2>/dev/null)"; echo "old: $(cat "$OLD.rev" 2>/dev/null)"
  echo "model: $MODEL"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; echo "load: $(uptime)"
} > "$OUT/provenance.txt"
python3 - "$OUT/plan.json" "$NEW" "$OLD" "$MODEL" "$IMG" "${ROUNDS:-4}" <<'PY'
import json, sys
out, new, old, model, img, rounds = sys.argv[1:]
def cell(name, serve, port=18657):
    return {"cell": name, "engine": "goinfer", "api": "openai", "port": port, "model": "", "media": "image", "base": img,
            "prompt": "Describe this image.",
            "cmd": [serve, "--model", model, "--backend", "webgpu", "--addr", f"127.0.0.1:{port}"]}
json.dump({"rounds": int(rounds), "per_round": 2, "rotate": True,
           "cells": [cell("qwen2.5-vl-3b-image-new", new), cell("qwen2.5-vl-3b-image-old", old)]}, open(out, "w"), indent=1)
PY
python3 -u "$B/vision_ttft.py" "$OUT/plan.json" "$OUT/qwen25vl-image" 2>&1 | tee "$OUT/qwen25vl-image.log"
rc=${PIPESTATUS[0]}
# Which route each arm's image prefill took, from its own log: the comparison is void if both took the same one.
grep -h -i "prefill" "$OUT"/qwen25vl-image/*.log 2>/dev/null | sort | uniq -c | head -20 > "$OUT/prefill-routes.txt"
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$OUT/provenance.txt"
exit "$rc"
