#!/usr/bin/env bash
# P26a's real-checkpoint gate on Qwen3.5-9B (TestQwen35VLReal_G2_9B: the 9B's image turn against HF f32, three images, near-tie rule) as an unattended night job.
# It took ~11 min in the v0.21.0 sweep (loads the 9B at f32, ~36 GB), so it is over the daytime limit. It tests REV (default origin/main) in its own detached worktree,
# counts a SKIP as a failure, and exits 2 on a missing asset. The 0.8B twin (TestQwen35VLReal_G2, ~22 s) and the serve gate (TestServe_qwen35Image_G4, ~45 s) ran by day.
#   python3 scripts/night.py add p26-9b-gate --est 20 --by "<who>, P26a" --doc docs/queue-performance.md -- bash docs/measurements/p26-batched-hybrid-image-prefill-2026-10-06/run-9b-gate.sh
#   TESTRE='^TestQwen35VLReal_G2$' bash .../run-9b-gate.sh   # the quick control-flow check
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
REV=${REV:-origin/main}
LOG=${1:-$HOME/goinfer-logs/p26-9b-gate-$(date +%F)}
WT=${WT:-$HOME/goinfer-bench/p26-9b-gate-wt}
TESTRE=${TESTRE:-^TestQwen35VLReal_G2_9B$}
export PATH=/usr/local/go/bin:$PATH
export GOINFER_HEAVY_TESTS=1
for p in "$HOME/models/qwen3.5-9b" "$HOME/models/qwen35vl_g2_9b" "$HOME/models/qwen3.5-0.8b"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing; the gate would skip"; exit 2; }
done
mkdir -p "$LOG" "$(dirname "$WT")"
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; testing the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
cd "$WT" || exit 2
{ echo "rev:      $(git rev-parse HEAD) ($REV)"; echo "started:  $(date '+%F %T %Z')"; echo "go:       $(go version)"; echo "test:     $TESTRE"; } | tee "$LOG/provenance.txt"
rc=0
GOWORK=off go test -tags realckpt ./decoder -count=1 -timeout 40m -run "$TESTRE" -v 2>&1 | tee "$LOG/gate.log" | grep -E '^(--- |PASS|FAIL|ok|panic)'
if ! grep -q -- "--- PASS: $(echo "$TESTRE" | tr -d '^$')" "$LOG/gate.log"; then echo "!! did not PASS (a skip or a failure; see $LOG/gate.log)"; rc=1; fi
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
