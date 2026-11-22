#!/usr/bin/env bash
# Runs the 36dbfbaf1 KV10 binary at depths 3900 and 65536, once on each tree's text (see run.sh).
set -uo pipefail
D=$HOME/goinfer-logs/relgate-2026-11-21/kv10-text
for r in 36dbfbaf1 e0d661712; do
  echo "== text from $r, start $(date '+%T')"
  cd "$HOME/goinfer-bench/kv10-text-$r/backend/cuda" || exit 2
  GOINFER_HEAVY_TESTS=1 GOINFER_KV10_DEPTHS=3900,65536 "$D/kv10.test" -test.run '^TestKV10_longContextFidelity$' -test.v -test.timeout 45m > "$D/text-$r.log" 2>&1
  grep -E 'kv3_longprefill_test.go:737|^--- ' "$D/text-$r.log"
done
