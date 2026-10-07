#!/usr/bin/env bash
# The CUDA half of the pre-tag GPU gate (`go run ./cmd/gate gpu`, RELEASING.md §C1-M names the Metal half) as an unattended night job. It tests REV (default
# origin/main, fetched at start) in its own detached worktree, so whatever is half-edited in the main checkout is not what runs. About 83 minutes: the part outside
# the heavy real-model tier took 4 min 54 s on 2026-10-06 (docs/measurements/gate-gpu-noheavy-2026-10-06/) and the heavy tier measured 78 min on 2026-09-28.
#
# Needs: an IDLE GPU (the gate checks), the timing lock (night.py takes it), the models under ~/models, and a go.work (gitignored, so the worktree gets one:
# the cuda module builds against THIS tree's root, not the last published tag). The gitignored fixtures are symlinked in from the main checkout, as the sweep does.
# A gate that did not print its own PASS verdict is a failure of this job, whatever the exit code: a skip is not a pass.
#   python3 scripts/night.py add gate-gpu-cuda --est 90 --by "<who>, R1" --doc docs/queue-release.md -- bash docs/measurements/gate-gpu-night-2026-10-06/run-gate-gpu.sh
#   GOINFER_GATE_SKIP_HEAVY=1 bash .../run-gate-gpu.sh /tmp/somewhere     # the 5-minute control of this script's own plumbing
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
REV=${REV:-origin/main}
LOG=${1:-$HOME/goinfer-logs/gate-gpu-night-$(date +%F)}
WT=${WT:-$HOME/goinfer-bench/gate-gpu-night-wt}
export PATH=/usr/local/go/bin:$PATH
[ -d "$HOME/models" ] || { echo "FATAL: $HOME/models is missing"; exit 2; }
mkdir -p "$LOG" "$(dirname "$WT")"
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; testing the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
n=0
for d in testdata decoder/testdata; do
  while IFS= read -r p; do
    p=${p%/}
    [ -e "$WT/$p" ] && continue
    if [ -d "$SRC/$p" ]; then
      mkdir -p "$WT/$p"
      for c in "$SRC/$p"/* "$SRC/$p"/.[!.]*; do [ -e "$c" ] && ln -s "$c" "$WT/$p/"; done
    else
      mkdir -p "$(dirname "$WT/$p")"
      ln -s "$SRC/$p" "$WT/$p"
    fi
    n=$((n + 1))
  done < <(git ls-files --others --ignored --exclude-standard --directory "$d")
done
cd "$WT" || exit 2
go work init . ./gpu ./cuda ./metal ./demo/agent || { echo "FATAL: go work init"; exit 2; }
{ echo "rev:      $(git rev-parse HEAD) ($REV)"; echo "started:  $(date '+%F %T %Z')"; echo "fixtures: $n gitignored entries symlinked from $SRC"
  echo "go:       $(go version)"; echo "gpu:      $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"
  echo "heavy:    $([ -n "${GOINFER_GATE_SKIP_HEAVY:-}" ] && echo SKIPPED || echo on)"; } | tee "$LOG/provenance.txt"
go run ./cmd/gate gpu -logdir "$LOG" 2>&1 | tee "$LOG/gate-gpu.log"
rc=${PIPESTATUS[0]}
clean=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG/gate-gpu.log")
if ! grep -q -- "PASS — cuda on Linux" <<<"$clean"; then echo "!! the gate printed no PASS verdict (rc=$rc); see $LOG/gate-gpu.log"; rc=1; fi
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
