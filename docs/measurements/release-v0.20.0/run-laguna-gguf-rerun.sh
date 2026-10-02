#!/usr/bin/env bash
# Re-run of TestLagunaGGUF_gate for v0.20.0, after a6ce3d4a. Sweep run 2 (bcf50a49, 2026-10-01) failed it at load: the
# fit guard refused it (56.5 GB priced against a 35.9 GB budget), on the box that passed it in the v0.19.0 sweep. The gate
# checks the loader's parity, so a6ce3d4a gave it the same per-model guard knob as TestQwen35GGUF_gate.
#
# Pass rule: the test's own verdict, "--- PASS: TestLagunaGGUF_gate", with nothing else changed (same box, same
# checkpoint from ~/models, realckpt tag). Runs on nobara-pc under the timing lock, after the Qwen3.5 bisect, since both
# hold tens of GB:
#
#   python3 scripts/timing_lock.py run --label release-v020-laguna-gguf-rerun -- bash run-laguna-gguf-rerun.sh
set -euo pipefail
REV=${REV:-f0cf6f2d}
SRC=$HOME/mycode/goinfer
BASE=$HOME/goinfer-bench/release-v0.20.0
WT=$BASE/laguna-wt
LOG=$HOME/goinfer-logs/release-v0.20.0/laguna-gguf-rerun.log
export PATH=/usr/local/go/bin:$PATH GOWORK=off GOINFER_HEAVY_TESTS=1
export GOINFER_LAGUNA_GGUF=$HOME/models/laguna-xs21-gguf/Laguna-XS-2.1-Q4_K_M.gguf

cd "$SRC"
git fetch -q origin
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$REV"
cd "$WT"
{
  echo "rev:      $(git rev-parse HEAD)"
  echo "started:  $(date '+%F %T %Z')"
  echo "go:       $(go version)"
  echo "asset:    $GOINFER_LAGUNA_GGUF"
} | tee "$LOG"
go test -count=1 -tags realckpt -run '^TestLagunaGGUF_gate$' -v -timeout 30m ./decoder/ 2>&1 | tee -a "$LOG"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG"
