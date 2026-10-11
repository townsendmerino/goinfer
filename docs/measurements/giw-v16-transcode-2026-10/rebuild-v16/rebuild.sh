#!/bin/bash
# Decision 4 of docs/completed/task-giw-v16-transcode-2026-10.md: rebuild the hand-built bundles at weights format v16,
# each from the source its old header names, with the settings its old file shows (int4, int8 embedding, canonical
# layout; Metal target for the one Metal bundle). New files are written under $B/new and nothing in ~/models or the
# archive is replaced here: the digests of old and new are printed for review first. Not a timed run.
set -uo pipefail
B=$HOME/goinfer-bench/v16-rebuild
M=$HOME/models
P=$B/prequant-v16-linux
echo "started $(date '+%Y-%m-%d %H:%M:%S %Z'); free on ~: $(df -h $HOME | tail -1 | awk '{print $4}')"
old() { echo "-- old $1"; "$B/giwdigest-v15" "$1" 2>&1; }
build() { # build <name> <source> <target>
  local out=$B/new/$1
  echo "== $1 from $(basename "$2") (target $3), started $(date '+%H:%M:%S')"
  if [ ! -e "$2" ]; then echo "SOURCE MISSING: $2"; return 1; fi
  find "$B/new" -name "$1*" -delete
  "$P" -quant int4 -target "$3" -o "$out" "$2" 2>&1 | tail -2
  [ -f "$out" ] || { echo "BUILD FAILED: $1"; return 1; }
  echo "-- new $1 ($(stat -c %s "$out") bytes)"; "$B/giwdigest-v16" "$out" 2>&1
}
old $M/qwen3.6-35b-a3b-int4.giw
build qwen3.6-35b-a3b-int4.giw $M/qwen3.6-35b-a3b-Q8_0.gguf canonical
old /srv/models/qwen3.6-35b-a3b-int4-v12.giw
build qwen3.6-35b-a3b-int4-v12.giw $M/qwen3.6-35b-a3b-q4_k_m.gguf canonical
echo "waiting for the source copies ..."; while [ -e /proc/$(cat $B/pull.pid) ]; do sleep 10; done; cat $B/pull.out
old $M/mellum2.int4.giw
build mellum2.int4.giw $M/mellum2-unq canonical
old $M/gemma4-26b-int4.giw
build gemma4-26b-int4.giw $M/gemma-4-26b-a4b-it canonical
old $M/gemma4-26b-int4-v14st.metal.giw
build gemma4-26b-int4-v14st.metal.giw $M/gemma-4-26b-a4b-it metal
echo "finished $(date '+%H:%M:%S'); free on ~: $(df -h $HOME | tail -1 | awk '{print $4}')"; ls -la $B/new | awk '{print $5, $9}'
