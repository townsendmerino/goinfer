#!/usr/bin/env bash
# The CUDA module's tests with the device present, as a night job (owner request, 2026-10-18). CI's cuda job runs on a runner with no GPU, where
# 329 of the module's tests skip (.github/workflows/coverage.yml); a test that fails only on a device (TestKernelLocalMemoryCensus, red on main from the
# commit that added mamba_conv until it was pinned) is invisible there. This runs `go test -json -tags 'cuda goinfer_testhooks' ./...` in ./cuda of REV
# (default origin/main, fetched at start) in its own detached worktree, with the fixtures placed, on nobara through the timing lock (night.py takes it).
# No self-hosted GitHub runner: the repo is public.
#
# The verdict is scripts/cuda_device_suite.py's, not go test's exit status: a failed test or package is red; a skip whose reason says no device is red;
# no passing test is red. The last lines of the log are the verdict and the failing names, which is what SUMMARY.md quotes.
#
# Expected wall time: about 5 min warm, 8 with a cold build cache. The device suite of the same tags took 12 s (kernel level) + 159 s (the testhooks suite) in
# docs/measurements/gate-gpu-noheavy-2026-10-06/gate-gpu-noheavy.log, and the whole noheavy gate 4 min 54 s. No heavy tier (GOINFER_HEAVY_TESTS is not set).
#   python3 scripts/night.py add cuda-device-suite --est 10 --by "<who>, device-only failures" --doc docs/measurements/cuda-device-suite-2026-10-18/README.md -- bash docs/measurements/cuda-device-suite-2026-10-18/run-cuda-device-suite.sh
#   REV=<sha> bash .../run-cuda-device-suite.sh [logdir]
#   TESTARGS='-run ^TestKernelLocalMemoryCensus$' bash .../run-cuda-device-suite.sh /tmp/x   # the plumbing control: worktree, fixtures, go.work, build, one device test (under a minute)
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
REV=${REV:-origin/main}
LOG=${1:-$HOME/goinfer-logs/cuda-device-suite-$(date +%F)}
WT=${WT:-$HOME/goinfer-bench/cuda-device-suite-wt}
export PATH=/usr/local/go/bin:$PATH
nvidia-smi -L >/dev/null 2>&1 || { echo "FATAL: no NVIDIA device; every device test would skip"; exit 2; }
mkdir -p "$LOG" "$(dirname "$WT")"
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; testing the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
EXC="$LOG/untracked-fixtures.exclude"
fixtures=$(bash "$SRC/scripts/worktree_fixtures.sh" "$SRC" "$WT" "$EXC") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }
cd "$WT" || exit 2
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0="$EXC"
go work init . ./gpu ./cuda ./metal ./demo/agent || { echo "FATAL: go work init"; exit 2; }
{ echo "rev:      $(git rev-parse HEAD) ($REV)"; echo "started:  $(date '+%F %T %Z')"; echo "fixtures: $fixtures"
  echo "go:       $(go version)"; echo "gpu:      $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; } | tee "$LOG/provenance.txt"
# The runner is the SRC checkout's copy, so a REV from before it existed can still be judged by it.
CGO_ENABLED=0 python3 "$SRC/scripts/cuda_device_suite.py" --cwd "$WT/cuda" --raw "$LOG/go-test.jsonl" -- \
  go test -json -count=1 -timeout 25m -tags 'cuda goinfer_testhooks' ${TESTARGS:-} ./... 2>&1 | tee "$LOG/verdict.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
