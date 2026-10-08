#!/bin/bash
# The re-run of the MacBook's vetted-checkpoint cells (docs/measurements/peer-vetted-2026-10-07-macbook.md, "The re-run",
# registered before it ran): run-mac.sh unchanged but for the harness, which runs from this script's own checkout (the
# calibrated G26Q:128 prompt, and Ollama's log kept beside goinfer's serve logs). Night queue only (the queue holds the timing lock).
# The cells: the vetted QAT Gemma 4 26B-A4B (M3) and gpt-oss 20B (M2) on Metal, depth 128, greedy, 3 runs, goinfer against Ollama 0.32.5, each at its
# defaults (goinfer paged: -moe-cache-experts, with -require-backend, the M26 day-use rule), context 2048 as cell c. One
# engine at a time; BENCH_SWAP_VOID_MB=0 voids an arm whose swap grew, BENCH_SWAP_KILL_MB=1024 kills a runaway server.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)   # this checkout: its scripts/bench_peer.py and scripts/prompts.json
REV=65b2c22a
B=$HOME/goinfer-bench/peer-vetted
OUT=$B/results-mac-2
R=$HOME/tmcode/goinfer
mkdir -p "$OUT"
[ -d "$B/wt-$REV" ] || git -C "$R" worktree add --detach "$B/wt-$REV" "$REV" || exit 1
[ "$(git -C "$B/wt-$REV" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
[ -x "$B/serve-metal-$REV" ] || (cd "$B/wt-$REV/metal" && CGO_ENABLED=0 go build -o "$B/serve-metal-$REV" ./cmd/serve) || exit 1
export GOINFER_SERVE_METAL=$B/serve-metal-$REV OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
{ echo "rev $REV (serve sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16))"; echo "ollama $($OLLAMA_BIN --version 2>&1 | tail -1)"
  sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; sysctl -n vm.swapusage; df -h ~ | tail -1; uptime; } | tee "$OUT/provenance.txt"
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama BENCH_DEPTHS=none BENCH_BACKENDS=metal BENCH_CTX=2048
export BENCH_QUANT_OVERRIDE="M35=int4mix,phi3-mini=default,G20=default,G26Q=default" BENCH_REQUIRE_BACKEND=1
export BENCH_SWAP_VOID_MB=0 BENCH_SWAP_KILL_MB=1024 BENCH_SERVE_LOG_DIR=$OUT/serve-logs
python3 -c "import json; json.load(open('$ROOT/scripts/prompts.json'))['G26Q:128']" || { echo "FATAL: no G26Q:128 prompt in $ROOT"; exit 2; }
cd "$ROOT" || exit 1
echo "=== $(date -u +%FT%TZ) START 26B on Metal"
BENCH_MODELS=G26Q python3 -u scripts/bench_peer.py "$OUT/metal-g26q.json"; rc=$?
echo "=== $(date -u +%FT%TZ) END rc=$rc; swap $(sysctl -n vm.swapusage)"
# M2 (the record's amendment): gpt-oss on Metal, the same cell shape, without -require-backend (its prefill is sequential on
# Metal, which that flag refuses and a decode rate does not use); the decode-path gate and the swap void/kill still apply.
echo "=== $(date -u +%FT%TZ) START gpt-oss on Metal"
BENCH_REQUIRE_BACKEND= BENCH_MODELS=G20 python3 -u scripts/bench_peer.py "$OUT/metal-g20.json"; rc2=$?
echo "=== $(date -u +%FT%TZ) END rc=$rc2; swap $(sysctl -n vm.swapusage)"
[ $rc -eq 0 ] && rc=$rc2
python3 "$HERE/grade.py" "$OUT" 2>&1 | tee "$OUT/grade.jsonl"
exit $rc
