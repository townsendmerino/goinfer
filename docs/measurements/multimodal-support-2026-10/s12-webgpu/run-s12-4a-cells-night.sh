#!/usr/bin/env bash
# S12.4a (docs/tasks/task-multimodal-support-2026-10.md, S12.4): S7's image and audio cells on the Mac under
# --backend webgpu, each family twice: with serve's defaults (every vision tower on the CPU) and with -vision-device gpu
# (the WebGPU towers). One warm-up and three timed requests per cell, each with media no server has seen
# (vision_ttft.py), TTFT against the 5 s bar. A record, not a gate; it says where an image turn's time goes on WebGPU
# before the image prefill is built there. night.py holds the timing lock for the job.
#
# Pinned: $BIN/serve-webgpu-s12-4 (rev in $BIN/serve-webgpu-s12-4.rev; go build -tags gpu ./cmd/serve in gpu/).
# Checkpoints from ~/models (never the archive).
# Estimate: 13 cells x (a load up to 1 min + 4 requests of up to a minute) ~ 55 min; queued at 80.
#   python3 scripts/night.py add s12-4a-webgpu-cells --est 80 --by "Claude (Mac), S12.4" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/s12-webgpu/run-s12-4a-cells-night.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s12-webgpu-2026-10-11/bin}
OUT=${1:-$HOME/goinfer-logs/s12-webgpu/4a-cells-$(date +%F)}
M=$HOME/models
[ -x "$BIN/serve-webgpu-s12-4" ] || { echo "FATAL: $BIN/serve-webgpu-s12-4 is missing"; exit 2; }
for d in gemma-3-4b-it gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf gemma-4-E2B-unq qwen25vl-3b-instruct qwen3.5-0.8b qwen3-vl-2b-instruct glm-ocr; do
  [ -e "$M/$d" ] || { echo "FATAL: ~/models/$d is missing"; exit 2; }
done
mkdir -p "$OUT"
IMG=$SRC/testdata/gemma3_preprocess_image.png
WAV=$SRC/testdata/speech/librispeech-1272-128104-0000.wav
[ -f "$IMG" ] && [ -f "$WAV" ] || { echo "FATAL: the test image or clip is missing"; exit 2; }
python3 - "$OUT/plan.json" "$BIN/serve-webgpu-s12-4" "$M" "$IMG" "$WAV" <<'PY'
import json, sys
out, serve, m, img, wav = sys.argv[1:]
def cell(name, args, media="image", port=18640):
    return {"cell": name, "engine": "goinfer", "api": "openai", "port": port, "model": "", "media": media,
            "base": wav if media == "audio" else img,
            "prompt": "Transcribe this audio." if media == "audio" else "Describe this image.",
            "cmd": [serve, *args, "--backend", "webgpu", "--addr", f"127.0.0.1:{port}"]}
e2b = ["--model", f"{m}/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf", "--vision", f"{m}/gemma-4-E2B-unq"]
fams = [("gemma-3-4b", ["--model", f"{m}/gemma-3-4b-it"]), ("gemma-4-e2b", e2b),
        ("qwen2.5-vl-3b", ["--model", f"{m}/qwen25vl-3b-instruct"]), ("qwen3.5-0.8b", ["--model", f"{m}/qwen3.5-0.8b"]),
        ("qwen3-vl-2b", ["--model", f"{m}/qwen3-vl-2b-instruct"]), ("glm-ocr", ["--model", f"{m}/glm-ocr"])]
cells = []
for name, args in fams:  # the two arms of a family back to back
    cells.append(cell(name + "-cpu-tower", args))
    cells.append(cell(name + "-webgpu-tower", args + ["-vision-device", "gpu"]))
cells.append(cell("gemma-4-e2b-audio-cpu-tower", e2b, media="audio"))
for c in cells:
    for a in c["cmd"]:
        if a.startswith("/Volumes/") or a.startswith("/srv/models"):
            sys.exit(f"{a} is the archive (CLAUDE.md)")
json.dump({"rounds": 1, "per_round": 3, "rotate": False, "cells": cells}, open(out, "w"), indent=1)
print(len(cells), "cells")
PY
{ echo "serve-webgpu-s12-4: main $(cat "$BIN/serve-webgpu-s12-4.rev" 2>/dev/null)"; echo "started: $(date '+%F %T %Z')"
  sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; uptime; vm_stat | head -5; } | tee "$OUT/provenance.txt"
cd "$SRC" && python3 docs/measurements/multimodal-support-2026-10/vision_ttft.py "$OUT/plan.json" "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
