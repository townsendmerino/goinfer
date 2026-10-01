#!/usr/bin/env bash
# RELEASING.md §C1 for v0.20.0, the scoped re-validation after aikit v1.51.1 (owner decision 2026-10-01: a scoped re-run,
# not a third full sweep). Sweep run 2 (bcf50a49) was green apart from two blockers. TestLagunaGGUF_gate was fixed in
# a6ce3d4a and passes (run-laguna-gguf-rerun.sh). TestQwen35GGUF_gate bisected to aikit v1.50.2's amd64 int8 quantizer,
# which rounded 0.49999997 to 1; aikit v1.51.1 fixes it, and with the fix the gate reproduced the pre-v1.50.2 result
# exactly (run-qwen35-gguf-roundfix.sh).
#
# The fix changes amd64 numerics wherever aikit's int8 quantizer runs: activation quantization before every W8A8/W4A8
# matmul, and int8 weight quantization at load (QuantizeRowInt8 / QuantizeRowsInt8 share the core). So this re-runs,
# on amd64 at REV (goinfer on aikit v1.51.1):
#   1. the whole untagged cell (./decoder/ ./tokenizer/, every tiny golden including the int4/int8 ones), REALCKPT=0;
#   2. every real-checkpoint gate whose source loads a non-f32 quant, directly or through realLogitOracle /
#      loadQwen35GGUFSlice (both int8int8): the 29 below. Left out: TestQwen35GGUF_vsSafetensors, which cannot hold
#      both of its models on this box and capacity-skips after loading the first (16 min in sweep run 2).
# The 26 real gates that load f32 never call the quantizer; their sweep-run-2 results stand.
#
# Pass rule, written before it runs: both invocations print the gate's own verdict "ALL REQUIRED GATES GREEN" for
# their scoped checkset, and TestQwen35GGUF_gate passes at argmax 68/80, cosine min 0.98740 / mean 0.99608 (the
# confirming run's numbers). About 1 h 55 min: ~37 min, then ~75 min, plus builds.
#
#   REV=<bump commit> python3 scripts/timing_lock.py run --label release-v020-scoped-revalidation -- bash run-scoped-revalidation.sh
set -euo pipefail
: "${REV:?set REV to the goinfer commit that pins aikit v1.51.1}"
SRC=$HOME/mycode/goinfer
BASE=$HOME/goinfer-bench/release-v0.20.0
WT=$BASE/scoped-wt
LOG=$HOME/goinfer-logs/release-v0.20.0/scoped-revalidation
export PATH=/usr/local/go/bin:$PATH
export GOINFER_QWEN35_REAL=${GOINFER_QWEN35_REAL:-/srv/models/qwen3.6-35b-a3b}
export GOINFER_QWEN38=${GOINFER_QWEN38:-/srv/models/qwen3.8-27b}
GATES=(TestDeepseekGGUFReal_gate TestGemma4_26B_gate TestGlm4MoeAir_gate TestGptOssReal_gate TestGptOssReal_logitParity
  TestGraniteReal_gate TestGraniteReal_oracle TestLagunaGGUF_gate TestLagunaReal_gate TestLagunaReal_oracle
  TestLlama4Real_gate TestNemotron35LightningReal_oracle TestNemotron3NanoMoEReal_gate TestNemotron3NanoReal_oracle
  TestNemotronReal_gate TestNemotronReal_oracle TestPhi3GGUFReal_gate TestQwen2MoeReal_oracle TestQwen35GGUF_gate
  TestQwen35GGUF_locateDivergence TestQwen35GGUF_routeFlipAtOutlier TestQwen35GGUF_weightDiff
  TestQwen35Real_gate2FullModel TestQwen38GGUF_gate TestQwen38GGUF_weightDiff TestQwen38Real_gate TestQwen38Real_oracle
  TestQwen3MoeReal_oracle TestQwen3NextReal_oracle)
[ "${#GATES[@]}" -eq 29 ]
RE="^($(IFS='|'; echo "${GATES[*]}"))\$"
mkdir -p "$BASE" "$LOG"

cd "$SRC"
git fetch -q origin
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$REV"
n=0
for d in testdata decoder/testdata; do
  while IFS= read -r p; do
    p=${p%/}
    [ -e "$WT/$p" ] && continue
    if [ -d "$SRC/$p" ]; then
      # a real directory holding symlinks, not a symlink to the directory (see run-parity-sweep.sh)
      mkdir -p "$WT/$p"
      for c in "$SRC/$p"/* "$SRC/$p"/.[!.]*; do [ -e "$c" ] && ln -s "$c" "$WT/$p/"; done
    else
      mkdir -p "$(dirname "$WT/$p")"
      ln -s "$SRC/$p" "$WT/$p"
    fi
    n=$((n + 1))
  done < <(git ls-files --others --ignored --exclude-standard --directory "$d")
done

cd "$WT"
grep -q 'github.com/townsendmerino/aikit v1.51.1' go.mod
{
  echo "rev:        $(git rev-parse HEAD) ($(grep 'townsendmerino/aikit v' go.mod | head -1 | xargs))"
  echo "started:    $(date '+%F %T %Z')"
  echo "fixtures:   $n gitignored entries symlinked from $SRC"
  echo "tree:       $(git status --porcelain | wc -l) entries in git status (0 = clean)"
  echo "go:         $(go version)"
  echo "archive:    GOINFER_QWEN35_REAL=$GOINFER_QWEN35_REAL GOINFER_QWEN38=$GOINFER_QWEN38"
  echo "gates:      ${#GATES[@]} real-checkpoint gates: $RE"
} | tee "$LOG/provenance.txt"

echo "== 1/2 untagged cell, all of it (REALCKPT=0) — $(date '+%T')" | tee -a "$LOG/provenance.txt"
GOWORK=off REALCKPT=0 EMIT_MANIFEST=1 go run ./cmd/gate parity 2>&1 | tee "$LOG/cell1.log" || true
echo "== 2/2 the ${#GATES[@]} quantized real-checkpoint gates (GATE_RUN) — $(date '+%T')" | tee -a "$LOG/provenance.txt"
GOWORK=off GATE_RUN="$RE" EMIT_MANIFEST=1 go run ./cmd/gate parity 2>&1 | tee "$LOG/realckpt.log" || true
cp testdata/parity_manifest.json "$LOG/parity_manifest.after.json"
git diff --stat >"$LOG/worktree-diff.txt" 2>&1 || true
echo "finished:   $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
