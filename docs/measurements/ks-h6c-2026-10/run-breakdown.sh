#!/bin/bash
# KS-H6 build reading: the post-pool allocations of a served build, per model (TestKSH6C_postPoolBreakdown). Run by night.py.
set -u
REV=${1:?usage: run-breakdown.sh <rev>}
WT=$HOME/goinfer-bench/ks-h6c-run-$REV
LOG=$HOME/goinfer-logs/ks-h6c-2026-10/breakdown-$(date +%Y-%m-%d-%H%M).log
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
cd "$HOME/mycode/goinfer" || exit 1
[ -d "$WT" ] || git worktree add -q --detach "$WT" "$REV" || exit 1
cd "$WT" || exit 1
printf 'go 1.27.0\n\ntoolchain go1.27.3\n\nuse (\n\t.\n\t./cuda\n\t./demo/agent\n\t./gpu\n\t./metal\n\t./sdktest\n)\n' > go.work
go test -c -tags 'cuda goinfer_testhooks' -o "$WT/cuda.test" ./cuda/ || { echo "FAILED: build"; exit 1; }
for spec in "qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf 16384" "qwen2.5-7b-instruct-q4_k_m.gguf 16384" "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf 0"; do
  set -- $spec
  echo "== $1 ctx $2 $(date +%H:%M:%S)"
  (cd "$WT/cuda" && KSH6C_MODEL=$HOME/models/$1 KSH6C_CTX=$2 "$WT/cuda.test" -test.run '^TestKSH6C_postPoolBreakdown$' -test.v -test.count 1 -test.timeout 15m 2>&1 | grep -E '^\[H6C\]|--- |FAIL|panic')
done
echo "DONE"
