#!/usr/bin/env bash
# Re-pin of testdata/olmo3_real_golden.json.gz under transformers 5.15.0, then TestOlmo3Real_gate against it.
# Why: the 2026-09-07 golden was pinned under transformers 5.12 (~/.venv-vl), whose Olmo3 applies the checkpoint's
# flat YaRN rope_scaling to every layer; 5.15 and the Olmo 3 paper (arXiv 2512.13961) put YaRN on full-attention layers
# only, which goinfer has done since 2aa4540b. Against the old golden the gate read cosine 0.992789.
#
# Pass rule, written before it runs: pin_olmo3_real.py's own RoPE check passes (finite per-layer-type tables, attention
# scaling 1.208 on full layers and 1.0 on sliding ones), and TestOlmo3Real_gate passes (its bars: argmax and the
# 8-token continuation exact, last-logit cosine >= 0.9999). Correctness, not timing; ~30 GB at f32, so it takes the
# timing lock. nobara-pc only (the checkpoint and the 5.15 venv live there).
#
#   REV=<commit> python3 scripts/timing_lock.py run --label olmo3-golden-repin -- bash run-repin.sh
set -euo pipefail
: "${REV:?set REV to the commit with the updated pin script}"
SRC=$HOME/mycode/goinfer
BASE=$HOME/goinfer-bench/olmo3-golden-repin-2026-10-01
WT=$BASE/wt
LOG=$HOME/goinfer-logs/olmo3-golden-repin-2026-10-01
PY=$HOME/.venv-triton-check/bin/python
export PATH=/usr/local/go/bin:$PATH GOWORK=off GOINFER_HEAVY_TESTS=1
export GOINFER_OLMO3_7B=$HOME/models/olmo3-7b-think
mkdir -p "$BASE" "$LOG"

cd "$SRC"
git fetch -q origin
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$REV"
cd "$WT"
{
  echo "rev:          $(git rev-parse HEAD)"
  echo "started:      $(date '+%F %T %Z')"
  echo "python:       $($PY -c 'import sys, torch, transformers; print(sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__)')"
  echo "go:           $(go version)"
  echo "checkpoint:   $GOINFER_OLMO3_7B"
} | tee "$LOG/provenance.txt"
echo "== pin — $(date '+%T')" | tee -a "$LOG/provenance.txt"
$PY scripts/pin_olmo3_real.py 2>&1 | tee "$LOG/pin.log"
cp testdata/olmo3_real_golden.json.gz "$LOG/"
echo "== gate — $(date '+%T')" | tee -a "$LOG/provenance.txt"
go test -count=1 -tags realckpt -run '^TestOlmo3Real_gate$' -v -timeout 30m ./decoder/ 2>&1 | tee "$LOG/gate.log" || true
echo "finished:     $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
