#!/usr/bin/env bash
# Confirms the mechanism the bisect found (run-bisect-qwen35-gguf.sh, 2026-10-01): the first bad commit is 81c250d7, the
# aikit v1.50.2 bump, whose amd64 AVX2 activation quantizer rounds y + copysign(0.5, y) and truncates. In float32 that
# sends |y| = 0.49999997 (nextafter32(0.5, 0)) to 1 where the scalar reference, math.Round, gives 0: shown on nobara-pc
# with a crafted row (aikit v1.50.1: 0 of 16 values differ; v1.51.0: exactly those two), and over every float32 with an
# exhaustive check (adding 0.5 misrounds 8,388,610 values, all but those two beyond the kernel's +-127 clamp; adding
# nextafter32(0.5, 0) misrounds none).
#
# This run: main (REV) with aikit v1.51.0 replaced by a copy patched in ONE constant (linalg/quant_act_amd64.s, the
# 0.5f broadcast becomes 0x3EFFFFFF). Pre-registered before it runs: if that rounding is the whole cause, the gate
# reproduces every GOOD step of the bisect EXACTLY: argmax 68/80, 7/10 prompts coherent, cosine min 0.98740, mean
# 0.99608, worst div gap 0.0080. Any other result means the rounding is not the whole story.
#
#   python3 scripts/timing_lock.py run --label release-v020-qwen35-gguf-roundfix -- bash run-qwen35-gguf-roundfix.sh
set -euo pipefail
REV=${REV:-f0cf6f2d}
SRC=$HOME/mycode/goinfer
BASE=$HOME/goinfer-bench/release-v0.20.0
FIX=${FIX:-$BASE/aikit-v1.51.0-roundfix}
WT=$BASE/roundfix-wt
LOG=$HOME/goinfer-logs/release-v0.20.0/qwen35-gguf-roundfix.log
export PATH=/usr/local/go/bin:$PATH GOWORK=off GOINFER_HEAVY_TESTS=1
export GOINFER_QWEN35_GGUF=$HOME/models/qwen3.6-35b-a3b-Q8_0.gguf
export GOINFER_QWEN35_GOLDEN=$HOME/models/qwen35_real_golden

grep -q 'MOVL    $0x3EFFFFFF, AX' "$FIX/linalg/quant_act_amd64.s" # the patched copy, not the module cache's
cd "$SRC"
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$REV"
cd "$WT"
go mod edit -replace "github.com/townsendmerino/aikit=$FIX"
{
  echo "rev:      $(git rev-parse HEAD) + go.mod replace aikit => $FIX"
  echo "patch:    $(grep -n '0x3EFFFFFF' "$FIX/linalg/quant_act_amd64.s")"
  echo "started:  $(date '+%F %T %Z')"
  echo "go:       $(go version)"
} | tee "$LOG"
go test -count=1 -tags realckpt -run '^TestQwen35GGUF_gate$' -v -timeout 30m ./decoder/ 2>&1 | tee -a "$LOG" || true
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG"
