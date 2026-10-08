#!/usr/bin/env bash
# S7 on nobara-pc, THIRD READ: Gemma 3 on the float32 default, Qwen2.5-VL without its 256 MiB slack, and Qwen3-VL-2B added as a cell (docs/tasks/task-multimodal-support-2026-10.md, "S7 third read, registered 2026-10-08"): every image and audio cell this box runs, on its best backend (CUDA), with
# serve's DEFAULT flags (the default image budget, the default KV plan): one warm-up and three timed requests per cell, each with media no server has seen (vision_ttft.py), TTFT
# against the 5 s bar. A record; the cells rank S16-S18.
#
# Pinned: $BIN/serve-cuda-s7r3 built from main (rev in $BIN/serve-cuda-s7r3.rev). Checkpoints from ~/models (never the archive). The queue holds the timing lock.
# Estimate: 7 cells x (load up to 1 min + 4 requests; Qwen3-VL-2B's CPU tower may take ~20 s a request) ~ 10 min; measured 2 min for the six cells of the second read.
#   python3 scripts/night.py add s7-nobara-3 --est 12 --by "nobara session, S7" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s7-nobara-3.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s7n}
OUT=${1:-$HOME/goinfer-bench/s7n/s7-nobara-3-$(date +%F)}
M=$HOME/models
CUR=$BIN/serve-cuda-s7r3
[ -x "$CUR" ] || { echo "FATAL: $CUR is missing"; exit 2; }
mkdir -p "$OUT"
IMG=$SRC/testdata/gemma3_preprocess_image.png
WAV=$SRC/testdata/speech/librispeech-1272-128104-0000.wav
for f in "$IMG" "$WAV"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
python3 - "$OUT/plan.json" "$CUR" "$M" "$IMG" "$WAV" <<'PY'
import json, os, sys
out, serve, m, img, wav = sys.argv[1:]
def cell(name, args, media="image", port=18640):
    return {"cell": name, "engine": "goinfer", "api": "openai", "port": port, "model": "", "media": media,
            "base": wav if media == "audio" else img,
            "prompt": "Transcribe this audio." if media == "audio" else "Describe this image.",
            "cmd": [serve, *args, "--backend", "cuda", "--addr", f"127.0.0.1:{port}"]}
e2b = ["--model", f"{m}/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf", "--vision", f"{m}/gemma-4-E2B-unq"]
cells = [cell("gemma-3-4b", ["--model", f"{m}/gemma-3-4b-it"]),
         cell("gemma-4-e2b", e2b), cell("gemma-4-e2b-audio", e2b, media="audio"),
         cell("qwen2.5-vl-3b", ["--model", f"{m}/qwen25vl-3b-instruct"]),
         cell("qwen3.5-0.8b", ["--model", f"{m}/qwen3.5-0.8b"]),
         cell("glm-ocr", ["--model", f"{m}/glm-ocr"]),
         # S10 landed on main (2026-10-08): serve takes Qwen3-VL images. CUDA has no DeepStack tap yet, so this cell reads what the CPU tower does.
         cell("qwen3-vl-2b", ["--model", f"{m}/qwen3-vl-2b-instruct"])]
# Third read: every cell runs on the current build ($CUR = serve-cuda-s7r3).
for c in cells:
    for a in c["cmd"]:
        if a.startswith("/Volumes/") or a.startswith("/srv/models"):
            sys.exit(f"{a} is the archive (CLAUDE.md)")
        if a.startswith(m) and not os.path.exists(a):
            sys.exit(f"{a} is missing")
json.dump({"rounds": 1, "per_round": 3, "rotate": False, "cells": cells}, open(out, "w"), indent=1)
print(len(cells), "cells")
PY
[ $? -eq 0 ] || exit 2
{ echo "serve-cuda-s7r3: $(cat "$BIN/serve-cuda-s7r3.rev" 2>/dev/null || echo unknown)"; echo "started: $(date '+%F %T %Z')"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; uname -sr; uptime; free -m | sed -n 2p; } | tee "$OUT/provenance.txt"
cd "$SRC" && python3 docs/measurements/multimodal-support-2026-10/vision_ttft.py "$OUT/plan.json" "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
