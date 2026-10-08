#!/usr/bin/env bash
# S13-lite on nobara-pc, the float32-tower addendum (docs/tasks/task-multimodal-support-2026-10.md, "S13-lite f32 arm, registered 2026-10-08 before it runs"): the same cell as run-s13lite-nobara.sh
# (Gemma 3 4B, a new image every request, 3 rounds of one warm-up and three timed requests per engine, the order rotated each round, one server at a time, every engine on CUDA) with one more
# goinfer arm, `-vision-quant f32`, which selects the float32 SigLIP tower on the tower base (2.07 s per image, S17 lever A) instead of the shipped default's int8 one (4.08 s). Four engines:
#   goinfer            serve's defaults (the int8 device tower)      -> the previous night's arm, repeated as this run's control
#   goinfer-f32        the same build with -vision-quant f32
#   ollama 0.32.5      gemma3:4b (blob sha256:aeda25e63ebd6...), its own server
#   llama.cpp 427291b  ggml-org Q4_K_M + mmproj-model-f16, CUDA
# goinfer is $BIN/serve-cuda-s18 (the build the previous night's S13-lite used; its rev is in provenance). A record, read against the 5 s bar and as ratios. Estimate: 4 engines x 3 rounds x
# (load + 4 requests) ~ 20 min; queued at 30.
#   python3 scripts/night.py add s13lite-f32-nobara --est 30 --by "nobara session, S13-lite f32 arm" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s13lite-f32-nobara.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s7n}
OUT=${1:-$HOME/goinfer-bench/s7n/s13lite-f32-nobara-$(date +%F)}
M=$HOME/models
OLLAMA=$HOME/ollama-0325/bin/ollama
LLAMA=${LLAMA:-$HOME/mycode/peers/llama.cpp/build/bin/llama-server}
GOINFER_BIN=$BIN/serve-cuda; [ -x "$BIN/serve-cuda-s18" ] && GOINFER_BIN=$BIN/serve-cuda-s18   # the current build (main + S18 + S17), amended 2026-10-07 evening before the run
for f in "$GOINFER_BIN" "$OLLAMA" "$M/gemma-3-4b-it/config.json" "$HOME/ollama-0325/models/manifests/registry.ollama.ai/library/gemma3/4b"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
for f in "$LLAMA" "$M/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf" "$M/gemma-3-4b-it-gguf/mmproj-model-f16.gguf"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
mkdir -p "$OUT"
python3 - "$OUT/plan.json" "$GOINFER_BIN" "$M" "$SRC/testdata/gemma3_preprocess_image.png" "$OLLAMA" "$LLAMA" <<'PY'
import json, sys
out, serve, m, img, ollama, llama = sys.argv[1:]
base = {"cell": "gemma-3-4b", "media": "image", "base": img, "prompt": "Describe this image."}
cells = [
    dict(base, engine="goinfer", api="openai", port=18650, model="",
         cmd=[serve, "--model", f"{m}/gemma-3-4b-it", "--backend", "cuda", "--addr", "127.0.0.1:18650"]),
    dict(base, engine="ollama", api="ollama", port=11535, model="gemma3:4b", cmd=[ollama, "serve"],
         env={"OLLAMA_HOST": "127.0.0.1:11535", "OLLAMA_MODELS": m.replace("/models", "/ollama-0325/models")}),
]
cells.insert(1, dict(base, engine="goinfer-f32", api="openai", port=18652, model="",
     cmd=[serve, "--model", f"{m}/gemma-3-4b-it", "--backend", "cuda", "-vision-quant", "f32", "--addr", "127.0.0.1:18652"]))
if True:
    cells.append(dict(base, engine="llama.cpp", api="openai", port=18651, model="",
         cmd=[llama, "-m", f"{m}/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf", "--mmproj", f"{m}/gemma-3-4b-it-gguf/mmproj-model-f16.gguf",
              "-ngl", "99", "--port", "18651", "--host", "127.0.0.1"]))
json.dump({"rounds": 3, "per_round": 3, "rotate": True, "cells": cells}, open(out, "w"), indent=1)
PY
{ echo "goinfer: $GOINFER_BIN (rev $(cat "${GOINFER_BIN}.rev" 2>/dev/null || echo unknown))"; echo "ollama: $(env -u LD_LIBRARY_PATH $OLLAMA --version 2>&1 | tail -1)"
  echo "ollama gemma3:4b model blob: $(python3 -c "import json,os;print([l['digest'] for l in json.load(open(os.path.expanduser('~/ollama-0325/models/manifests/registry.ollama.ai/library/gemma3/4b')))['layers'] if l['mediaType'].endswith('model')][0])")"
  echo "llama-server: $($LLAMA --version 2>&1 | head -1) at $LLAMA"; echo "started: $(date '+%F %T %Z')"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; uname -sr; uptime; } | tee "$OUT/provenance.txt"
cd "$SRC" && env -u LD_LIBRARY_PATH python3 docs/measurements/multimodal-support-2026-10/vision_ttft.py "$OUT/plan.json" "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
