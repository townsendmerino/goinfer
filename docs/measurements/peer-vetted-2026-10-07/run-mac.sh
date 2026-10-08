#!/bin/bash
# Vetted-checkpoint peer cells on the MacBook (M1 Pro 16 GB), pre-registered in
# docs/measurements/peer-vetted-2026-10-07-macbook.md before this ran. Night queue only (the queue holds the timing lock).
# The cell: the vetted QAT Gemma 4 26B-A4B on Metal, depth 128, greedy, 3 runs, goinfer against Ollama 0.32.5, each at its
# defaults (goinfer paged: -moe-cache-experts, with -require-backend, the M26 day-use rule), context 2048 as cell c. One
# engine at a time; BENCH_SWAP_VOID_MB=0 voids an arm whose swap grew, BENCH_SWAP_KILL_MB=1024 kills a runaway server.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REV=65b2c22a
B=$HOME/goinfer-bench/peer-vetted
OUT=$B/results-mac-$(date +%F)
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
cd "$B/wt-$REV" || exit 1
echo "=== $(date -u +%FT%TZ) START 26B on Metal"
BENCH_MODELS=G26Q python3 -u scripts/bench_peer.py "$OUT/metal-g26q.json"; rc=$?
echo "=== $(date -u +%FT%TZ) END rc=$rc; swap $(sysctl -n vm.swapusage)"
python3 "$HERE/grade.py" "$OUT" 2>&1 | tee "$OUT/grade.jsonl"
exit $rc
