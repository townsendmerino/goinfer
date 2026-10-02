#!/usr/bin/env bash
# Re-run of TestQwen38GGUF_gate for v0.20.0, after its fit-guard knob. It passed in sweep run 2 (bcf50a49, 114.9 s)
# and was refused at load in the scoped re-validation's step 2 (70be7081, 2026-10-01): the guard priced 66.6 GB
# (19.2 GB int4 + 32.0 GB KV at the model's full context + 15.3 GB for the mapped .gguf) against the 29.5 GB budget
# left after the 35B gates before it. The gate checks the loader's parity, so it now takes the same per-model guard
# knob as TestQwen35GGUF_gate and TestLagunaGGUF_gate.
#
# Pass rule: the test's own verdict, "--- PASS: TestQwen38GGUF_gate", with nothing else changed (same box, same
# checkpoint from ~/models, realckpt tag), at REV (aikit v1.51.1).
#
#   REV=<commit> python3 scripts/timing_lock.py run --label release-v020-qwen38-gguf-rerun -- bash run-qwen38-gguf-rerun.sh
set -euo pipefail
: "${REV:?set REV to the commit with the knob}"
SRC=$HOME/mycode/goinfer
BASE=$HOME/goinfer-bench/release-v0.20.0
WT=$BASE/qwen38-wt
LOG=$HOME/goinfer-logs/release-v0.20.0/qwen38-gguf-rerun.log
export PATH=/usr/local/go/bin:$PATH GOWORK=off GOINFER_HEAVY_TESTS=1
export GOINFER_QWEN38_GGUF=$HOME/models/qwen38-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf

cd "$SRC"
git fetch -q origin
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$REV"
cd "$WT"
{
  echo "rev:      $(git rev-parse HEAD)"
  echo "started:  $(date '+%F %T %Z')"
  echo "go:       $(go version)"
  echo "asset:    $GOINFER_QWEN38_GGUF"
} | tee "$LOG"
go test -count=1 -tags realckpt -run '^TestQwen38GGUF_gate$' -v -timeout 30m ./decoder/ 2>&1 | tee -a "$LOG"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG"
