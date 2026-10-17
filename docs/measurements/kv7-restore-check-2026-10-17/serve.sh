#!/usr/bin/env bash
# Start the pilot's server (binary and flags of pilot-server-352fba0.toml) on 43299. $1 = log path; extra args follow (e.g. -kv-host-bytes 0).
LOG=$1; shift
exec ~/tack-runs/bin/goinfer-serve-cuda-0e598d3f -model ~/models/qwen2.5-7b-instruct-q4_k_m.gguf -backend cuda -kv-sessions 1 -ctx 32768 -log-requests -served-model-name qwen -addr 127.0.0.1:43299 -log-token-ids "$LOG.ids" "$@" > "$LOG" 2>&1
