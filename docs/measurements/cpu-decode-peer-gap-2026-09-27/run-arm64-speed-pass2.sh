#!/bin/bash
# L1 gate 6 (arm64 CPU speed), PASS 2 — REVERSED ORDER. Pre-registered 2026-09-28 07:17 PDT (14:17 UTC), after pass 1's
# goinfer 0.5B (57.3) and ollama 0.5B (130.8) cells had landed and BEFORE any goinfer_old cell existed, so before any
# new ÷ old ratio was visible.
#
# Why: docs/measurements/r13-cpu-depth-row-2026-09-19.md Finding 2 — on this MacBook the same goinfer CPU cell halved
# (99.2 -> 48.9 tok/s) within ~18 minutes of sustained load while loadavg stayed flat (probably P-core thermal
# throttling; unverified, and `pmset -g therm` records nothing and powermetrics needs root, so there is still no
# thermal telemetry). bench_peer.py's cell order is fixed per model: goinfer -> ollama -> goinfer_old. In pass 1 that
# is new -> ollama -> old, so any monotone drift lands on the ratio.
#
# Design: pass 2 swaps the BINARIES behind the labels, so its order per model is old -> ollama -> new. In
# arm64-speed-served-pass2.json the label "goinfer" is the OLD build (3cd62e6d) and "goinfer_old" is the NEW build
# (5c85f7c0). Everything else as pass 1.
#
# Rule, per model:
#   - reported new ÷ old = geometric mean of pass 1's and pass 2's new ÷ old (a linear drift across a model's three
#     cells cancels);
#   - if the passes disagree on direction (one >= 1.03, the other <= 0.97), that model is reported UNRESOLVED BY DRIFT,
#     with both ratios, and no direction is claimed;
#   - the NEON-widen (FCVTL) follow-up the task names is flagged when the combined ratio is < 1.00 and both passes are
#     below 1.00.
# Reported, not a ship band: the owner decides the arm64 default.
set -u
L=$HOME/goinfer-bench/l1-mac
until grep -q "^=== .* END rc=" $L/arm64-speed.log; do sleep 20; done
cd /Users/francistownsend-merino/tmcode/goinfer
export GOINFER_SERVE_CPU=$L/serve-cpu-3cd62e6d GOINFER_SERVE_CPU_OLD=$L/serve-cpu-5c85f7c0   # SWAPPED on purpose
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old,ollama BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_MODELS=0.5B,1.5B,7B
export BENCH_MAX_LOADAVG=2.5 BENCH_IDLE_WAIT=1800
for i in $(seq 1 90); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l" 'BEGIN{exit !(l<=2.5)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done
echo "=== $(date '+%F %T %Z') / $(date -u '+%T UTC') START L1 gate 6 arm64 served PASS 2 (reversed: goinfer=OLD 3cd62e6d, goinfer_old=NEW 5c85f7c0)"
sysctl vm.swapusage
python3 scripts/bench_peer.py $L/arm64-speed-served-pass2.json; rc=$?
sysctl vm.swapusage
echo "=== $(date '+%F %T %Z') / $(date -u '+%T UTC') END PASS 2 rc=$rc"
