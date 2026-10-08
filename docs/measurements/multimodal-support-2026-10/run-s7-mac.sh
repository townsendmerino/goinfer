#!/usr/bin/env bash
# S7 on the Mac (docs/tasks/task-multimodal-support-2026-10.md, "S7 and S13-lite, registered"): every image and audio cell
# this box runs, on its best backend (Metal), with serve's DEFAULT flags (the default image budget, the default KV plan):
# one warm-up and three timed requests per cell, each with media no server has seen (vision_ttft.py), TTFT against the
# 5 s bar. A record; the cells rank S16-S18.
#
# Pinned: $BIN/serve-metal built from s2-towers (rev in $BIN/serve-metal.rev). Checkpoints from ~/models (never the archive).
# Estimate: 8 cells x (load up to 1 min + 4 requests of up to 30 s) ~ 25 min; queued at 40.
#   python3 scripts/night.py add s7-mac --est 40 --by "Claude, S7" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s7-mac.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s7}
OUT=${1:-$HOME/goinfer-bench/s7/s7-mac-$(date +%F)}
M=$HOME/models
[ -x "$BIN/serve-metal" ] || { echo "FATAL: $BIN/serve-metal is missing"; exit 2; }
mkdir -p "$OUT"
IMG=$SRC/testdata/gemma3_preprocess_image.png
WAV=$SRC/testdata/speech/librispeech-1272-128104-0000.wav
python3 - "$OUT/plan.json" "$BIN/serve-metal" "$M" "$IMG" "$WAV" <<'PY'
import json, os, sys
out, serve, m, img, wav = sys.argv[1:]
def cell(name, args, media="image", port=18610):
    return {"cell": name, "engine": "goinfer", "api": "openai", "port": port, "model": "", "media": media,
            "base": wav if media == "audio" else img,
            "prompt": "Transcribe this audio." if media == "audio" else "Describe this image.",
            "cmd": [serve, *args, "--backend", "metal", "--addr", f"127.0.0.1:{port}"]}
e2b = ["--model", f"{m}/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf", "--vision", f"{m}/gemma-4-E2B-unq"]
cells = [cell("gemma-3-4b", ["--model", f"{m}/gemma-3-4b-it"]),
         cell("gemma-4-e2b", e2b), cell("gemma-4-e2b-audio", e2b, media="audio"),
         cell("qwen2.5-vl-3b", ["--model", f"{m}/qwen25vl-3b-instruct"]),
         cell("qwen3.5-0.8b", ["--model", f"{m}/qwen3.5-0.8b"]),
         cell("qwen3-vl-2b", ["--model", f"{m}/qwen3-vl-2b-instruct"]),
         cell("glm-ocr", ["--model", f"{m}/glm-ocr"])]
giw = f"{m}/gemma-4-E4B-it.int4.metal.giw"
if os.path.exists(giw):  # built by tonight's s6-e4b job when it ran first
    cells.append(cell("gemma-4-e4b", ["--model", giw, "--vision", f"{m}/gemma-4-E4B-it"]))
for c in cells:
    for a in c["cmd"]:
        if a.startswith("/Volumes/") or a.startswith("/srv/models"):
            sys.exit(f"{a} is the archive (CLAUDE.md)")
json.dump({"rounds": 1, "per_round": 3, "rotate": False, "cells": cells}, open(out, "w"), indent=1)
print(len(cells), "cells")
PY
{ echo "serve-metal: s2-towers $(cat "$BIN/serve-metal.rev" 2>/dev/null)"; echo "started: $(date '+%F %T %Z')"
  sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; uptime; vm_stat | head -5; } | tee "$OUT/provenance.txt"
cd "$SRC" && python3 docs/measurements/multimodal-support-2026-10/vision_ttft.py "$OUT/plan.json" "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
