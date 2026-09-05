#!/usr/bin/env bash
# POST-FIX re-run of the W3 depth-8000 goinfer cells for M35/M26, after the batched-CUDA-prefill
# fix for MoE models (4ee59e15/654fa481/a9c23c67/31a2f589/a83fdbd4/5cc48545 -- "MoE models take the
# batched prefill path (P20 blocker 3) -- bit-identical, 1.08x"). The original run
# (nobara-w1-d7-m35-m26_2026-09-04.json) measured goinfer M35/M26 at depth 8000 taking 1528.8s /
# 384.3s wall clock (SEQUENTIAL prefill, one forward per prompt token) against llama.cpp's 61.7s /
# 67.0s -- this run repeats ONLY the goinfer cells (Ollama/llama.cpp are unaffected by this fix and
# already correct in the original file) to see how much of that wall-clock gap the fix closes.
# Writes to a NEW file -- the original run is not touched.
#
# Binary pinned to 9453430e, NOT the earlier 654fa481/2bae03fc: 9453430e makes the CUDA batched
# prefill path refuse Gated-DeltaNet (M35) explicitly (`r.dnet != nil`) instead of declining only
# by accident (an unrelated empty-string sentinel that happened to trip for the same model). M35's
# computed path is unchanged either way -- this is a provenance/correctness-hygiene pin, not a
# behavior difference for the numbers below. An earlier attempt at this same re-run was aborted
# after 2 of 4 cells (both depth-128, cheap) against a binary built at 2bae03fc, precisely so the
# expensive depth-8000 cells would be measured against the corrected-guard binary instead.
# Same reduced-cost protocol as the original: BENCH_RUNS left at the harness deep-cell default
# (ncomp=2, nruns=2, ngen=128).
set -euo pipefail
cd /home/francis/mycode/goinfer
export PATH=$HOME/.local/bin:/usr/local/go/bin:$HOME/go/bin:$PATH
export GOINFER_SERVE_CUDA=/home/francis/bench-current/serve-cuda-9453430e
export BENCH_MODELS=M35,M26
export BENCH_BACKENDS=cuda
export BENCH_ENGINES=goinfer
export BENCH_DEPTHS=8000
export BENCH_DEEP_CTX=8192
exec python3 scripts/bench_peer.py docs/measurements/peer-matrix-2026-09/nobara-w3-8k-m35-m26-postfix_2026-09-05.json
