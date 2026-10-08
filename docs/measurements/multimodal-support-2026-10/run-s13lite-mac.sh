#!/usr/bin/env bash
# S13-lite on the Mac (docs/tasks/task-multimodal-support-2026-10.md, "S7 and S13-lite, registered"): Gemma 3 4B against
# Ollama and llama.cpp, a new image every request (vision_ttft.py), three rounds of one warm-up and three timed requests
# per engine, the engine order rotated each round (same-session interleaved, one server at a time: three 4B servers do
# not fit 16 GB together). Every engine on Metal at its own defaults. A record, read against the 5 s bar and as a ratio.
#   goinfer:   $BIN/serve-metal (s2-towers, rev in serve-metal.rev), ~/models/gemma-3-4b-it (HF safetensors, int4 at load)
#   Ollama:    /opt/homebrew/bin/ollama 0.32.5, gemma3:4b (model blob sha256:aeda25e63ebd6...), its own server on 11535
#   llama.cpp: /opt/homebrew/bin/llama-server build 10621 (c1d0e7a00), ggml-org/gemma-3-4b-it-GGUF Q4_K_M + mmproj-model-f16
# Estimate: 3 engines x 3 rounds x (load up to 1 min + 4 requests) ~ 25 min; queued at 40.
#   python3 scripts/night.py add s13lite-mac --est 40 --by "Claude, S13-lite" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s13lite-mac.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s7}
OUT=${1:-$HOME/goinfer-bench/s7/s13lite-mac-$(date +%F)}
M=$HOME/models
OLLAMA=/opt/homebrew/bin/ollama LLAMA=/opt/homebrew/bin/llama-server
for f in "$BIN/serve-metal" "$OLLAMA" "$LLAMA"; do [ -x "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
for f in "$M/gemma-3-4b-it/config.json" "$M/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf" "$M/gemma-3-4b-it-gguf/mmproj-model-f16.gguf"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
mkdir -p "$OUT"
python3 - "$OUT/plan.json" "$BIN/serve-metal" "$M" "$SRC/testdata/gemma3_preprocess_image.png" "$OLLAMA" "$LLAMA" <<'PY'
import json, sys
out, serve, m, img, ollama, llama = sys.argv[1:]
base = {"cell": "gemma-3-4b", "media": "image", "base": img, "prompt": "Describe this image."}
cells = [
    dict(base, engine="goinfer", api="openai", port=18620, model="",
         cmd=[serve, "--model", f"{m}/gemma-3-4b-it", "--backend", "metal", "--addr", "127.0.0.1:18620"]),
    dict(base, engine="ollama", api="ollama", port=11535, model="gemma3:4b", cmd=[ollama, "serve"],
         env={"OLLAMA_HOST": "127.0.0.1:11535"}),
    dict(base, engine="llama.cpp", api="openai", port=18621, model="",
         cmd=[llama, "-m", f"{m}/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf", "--mmproj",
              f"{m}/gemma-3-4b-it-gguf/mmproj-model-f16.gguf", "-ngl", "99", "--port", "18621", "--host", "127.0.0.1"]),
]
json.dump({"rounds": 3, "per_round": 3, "rotate": True, "cells": cells}, open(out, "w"), indent=1)
PY
{ echo "serve-metal: s2-towers $(cat "$BIN/serve-metal.rev" 2>/dev/null)"; echo "ollama: $($OLLAMA --version 2>&1 | tail -1)"
  echo "llama-server: $($LLAMA --version 2>&1 | head -1)"; echo "started: $(date '+%F %T %Z')"
  sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; uptime; } | tee "$OUT/provenance.txt"
cd "$SRC" && python3 docs/measurements/multimodal-support-2026-10/vision_ttft.py "$OUT/plan.json" "$OUT" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
