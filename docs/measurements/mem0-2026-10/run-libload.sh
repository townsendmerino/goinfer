#!/bin/bash
# MEM0 half D: the library path. decoder.Load on a .gguf, no sidecar, one process per load. Expect about 1 to 2 minutes.
cd /home/francis/wt/serve-main || exit 1
go build -o /tmp/libload-mem0 docs/measurements/mem0-2026-10/testdata/libload.go || exit 1
for spec in "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf " "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf int4" "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf int4" "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf "; do
  set -- $spec
  /tmp/libload-mem0 "$HOME/models/$1" "${2:-}"
done
