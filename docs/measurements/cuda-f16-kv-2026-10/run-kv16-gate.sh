#!/usr/bin/env bash
# The fidelity gate of docs/tasks/task-cuda-f16-kv-2026-10.md as an unattended night job: cuda.TestKV16GateVsReference on the D7 decision cell (K=8000) and the S
# confirmation cell (K=3900), arms -kv f32 and -kv f16, set B, overflow survey inside the f32 arm. Estimated 30 minutes: the flash-decode gate's D7@8000 took 1,110 s for
# the same count of prompt runs (21), the survey adds a K/V download per layer per prompt, S@3900 takes about a minute (an exploratory run of it took 46 s).
#
# REV is required: the commit under test, pinned in its own detached worktree so a push during the night does not change what runs. A cross-module test needs a go.work
# (gitignored, so a fresh worktree has none); this one is written here, with the local aikit checkout the main go.work also uses, and its HEAD recorded.
#   python3 scripts/night.py add kv16-gate --est 30 --by "<who>, CUDA f16 KV" --doc docs/tasks/task-cuda-f16-kv-2026-10.md -- env REV=<sha> bash docs/measurements/cuda-f16-kv-2026-10/run-kv16-gate.sh
#
# Pass rule (pre-registered, the task doc): both cells print "=> PASSES" on their "=== KV16 GATE" line and the overflow survey prints under 32752. A SKIP is a failure.
set -uo pipefail
REV=${REV:?REV is required: the commit under test, never a default}
SRC=$HOME/mycode/goinfer
AIKIT=${AIKIT:-$HOME/mycode/aikit/aikit}
WT=${WT:-$HOME/goinfer-bench/kv16-gate-wt}
LOG=${LOG:-$HOME/goinfer-logs/kv16/gate-$(date +%F)}
export PATH=/usr/local/go/bin:$PATH
case "$WT $LOG" in *"/srv/models"*|*"/Volumes/"*) echo "FATAL: an archive path"; exit 2;; esac
for p in "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/goinfer-logs/prefill-ref-b/D7-K8000-p9.bin" "$HOME/goinfer-logs/prefill-ref-b/S-K3900-p9.bin"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing; its cell would skip"; exit 2; }
done
mkdir -p "$LOG" "$(dirname "$WT")"
cd "$SRC" || exit 2
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
cd "$WT" || exit 2
printf 'go 1.27.0\n\ntoolchain go1.27.2\n\nuse (\n\t.\n\t./cuda\n\t./gpu\n\t%s\n)\n' "$AIKIT" > go.work
{ echo "rev:      $(git rev-parse HEAD)"; echo "aikit:    $(git -C "$AIKIT" rev-parse HEAD)"; echo "started:  $(date '+%F %T %Z')"; echo "go:       $(go version)"
  echo "driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"; } | tee "$LOG/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the worktree and the provenance block"; cd "$SRC" && git worktree remove --force "$WT"; exit 0; }
cd cuda || exit 2
GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 \
  go test -tags 'cuda goinfer_testhooks' -count=1 -run '^TestKV16GateVsReference$' -v -timeout 100m . 2>&1 | tee "$LOG/kv16-gate.log" | grep -E '^(\[kv16|===|--- |\s+--- |PASS|FAIL|ok|panic)|PRECONDITION'
rc=${PIPESTATUS[0]}
for cell in "D7 K=8000" "S K=3900"; do
  grep -q "=== KV16 GATE $cell .* => PASSES" "$LOG/kv16-gate.log" || { echo "!! $cell did not print PASSES (a skip, a void precondition or a fail; see $LOG/kv16-gate.log)"; rc=1; }
done
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
