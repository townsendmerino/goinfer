#!/usr/bin/env bash
# S13-lite on nobara-pc (docs/tasks/task-multimodal-support-2026-10.md, "S7 and S13-lite, nobara half"): Gemma 3 4B against Ollama, a new image every request (vision_ttft.py),
# three rounds of one warm-up and three timed requests per engine, the engine order rotated each round (same-session interleaved, one server at a time: two 4B servers do not fit
# the 8 GB card together). Every engine on CUDA at its own defaults. A record, read against the 5 s bar and as a ratio.
#   goinfer: $BIN/serve-cuda (main, rev in serve-cuda.rev), ~/models/gemma-3-4b-it (HF safetensors, int4 at load), serve's defaults
#   Ollama:  ~/ollama-0325/bin/ollama 0.32.5, gemma3:4b (model blob sha256:aeda25e63ebd6..., the same blob the Mac's cell reads), its own server on 11535, OLLAMA_MODELS=~/ollama-0325/models
#   llama.cpp: NOT run on this box: no cmake and no nvcc here, and no llama-server build. The script REFUSES to run unless S13_NO_LLAMA=1 is set, so the missing arm is a stated choice and
#   never a silent drop. If LLAMA=<llama-server built with CUDA> is set, it is added as the third engine (ggml-org/gemma-3-4b-it-GGUF Q4_K_M + mmproj-model-f16, as the Mac's).
# Estimate: 2 engines x 3 rounds x (load up to 1 min + 4 requests, goinfer's CPU-decode ones up to ~1 min each) ~ 30 min; queued at 45.
#   S13_NO_LLAMA=1 python3 scripts/night.py add s13lite-nobara --est 45 --by "nobara session, S13-lite" --doc docs/tasks/task-multimodal-support-2026-10.md -- env S13_NO_LLAMA=1 bash docs/measurements/multimodal-support-2026-10/run-s13lite-nobara.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s7n}
OUT=${1:-$HOME/goinfer-bench/s7n/s13lite-nobara-$(date +%F)}
M=$HOME/models
OLLAMA=$HOME/ollama-0325/bin/ollama
LLAMA=${LLAMA:-}
if [ -z "$LLAMA" ] && [ "${S13_NO_LLAMA:-}" != 1 ]; then echo "FATAL: no llama.cpp on this box; set S13_NO_LLAMA=1 to run without it (a stated choice), or LLAMA=<llama-server>"; exit 2; fi
for f in "$BIN/serve-cuda" "$OLLAMA" "$M/gemma-3-4b-it/config.json" "$HOME/ollama-0325/models/manifests/registry.ollama.ai/library/gemma3/4b"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
[ -z "$LLAMA" ] || for f in "$LLAMA" "$M/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf" "$M/gemma-3-4b-it-gguf/mmproj-model-f16.gguf"; do [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
mkdir -p "$OUT"
python3 - "$OUT/plan.json" "$BIN/serve-cuda" "$M" "$SRC/testdata/gemma3_preprocess_image.png" "$OLLAMA" "$LLAMA" <<'PY'
import json, sys
out, serve, m, img, ollama, llama = sys.argv[1:]
base = {"cell": "gemma-3-4b", "media": "image", "base": img, "prompt": "Describe this image."}
cells = [
    dict(base, engine="goinfer", api="openai", port=18650, model="",
         cmd=[serve, "--model", f"{m}/gemma-3-4b-it", "--backend", "cuda", "--addr", "127.0.0.1:18650"]),
    dict(base, engine="ollama", api="ollama", port=11535, model="gemma3:4b", cmd=[ollama, "serve"],
         env={"OLLAMA_HOST": "127.0.0.1:11535", "OLLAMA_MODELS": m.replace("/models", "/ollama-0325/models")}),
]
if llama:
    cells.append(dict(base, engine="llama.cpp", api="openai", port=18651, model="",
         cmd=[llama, "-m", f"{m}/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf", "--mmproj", f"{m}/gemma-3-4b-it-gguf/mmproj-model-f16.gguf",
              "-ngl", "99", "--port", "18651", "--host", "127.0.0.1"]))
json.dump({"rounds": 3, "per_round": 3, "rotate": True, "cells": cells}, open(out, "w"), indent=1)
PY
{ echo "serve-cuda: main $(cat "$BIN/serve-cuda.rev" 2>/dev/null)"; echo "ollama: $(env -u LD_LIBRARY_PATH $OLLAMA --version 2>&1 | tail -1)"
  echo "ollama gemma3:4b model blob: $(python3 -c "import json,os;print([l['digest'] for l in json.load(open(os.path.expanduser('~/ollama-0325/models/manifests/registry.ollama.ai/library/gemma3/4b')))['layers'] if l['mediaType'].endswith('model')][0])")"
  echo "llama-server: ${LLAMA:-not run (no cmake/nvcc on this box)}"; echo "started: $(date '+%F %T %Z')"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; uname -sr; uptime; } | tee "$OUT/provenance.txt"
cd "$SRC" && env -u LD_LIBRARY_PATH python3 docs/measurements/multimodal-support-2026-10/vision_ttft.py "$OUT/plan.json" "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
