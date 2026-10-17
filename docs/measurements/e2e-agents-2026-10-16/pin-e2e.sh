#!/bin/bash
# Pin the end-to-end agent run (README.md) by day, before queueing: build serve at REV from a detached worktree and copy the matrix
# harness at REV plus this directory's plan, runner and extractor beside it, so the night job runs what was pinned whatever the tree does.
#
#   REV=<commit> B=$HOME/goinfer-bench/e2e-agents-2026-10-16 bash pin-e2e.sh
#
# Writes $B/bin/serve-cuda-e2e (+ .rev, .sha256), $B/run-e2e.sh, $B/matrix-pinned/ (matrix at REV, plan-e2e.json with serve and commit
# filled in, e2e_metrics.py) and $B/pinned.txt. Needs no GPU.
set -euo pipefail
REV=${REV:?the commit to pin}
B=${B:?the pinned directory}
REPO=${REPO:-$HOME/mycode/goinfer}
HERE=docs/measurements/integration-harness-2026-10
SRC=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B/bin"
WT="$B/wt"
if [ ! -d "$WT" ]; then
  git -C "$REPO" worktree add --detach "$WT" "$REV" >/dev/null
fi
git -C "$WT" checkout -q --detach "$REV"
FULL=$(git -C "$WT" rev-parse HEAD)
cat > "$WT/go.work" <<EOF
go 1.27.0

toolchain go1.27.2

use (
	.
	./cuda
	./gpu
)
EOF
(cd "$WT" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-cuda-e2e" github.com/townsendmerino/goinfer/cuda/cmd/serve)
echo "$FULL" > "$B/bin/serve-cuda-e2e.rev"
sha256sum "$B/bin/serve-cuda-e2e" | cut -d' ' -f1 > "$B/bin/serve-cuda-e2e.sha256"
rm -rf "$B/matrix-pinned"
cp -r "$WT/$HERE/matrix" "$B/matrix-pinned"
cp "$SRC/e2e_metrics.py" "$B/matrix-pinned/"
cp "$SRC/run-e2e.sh" "$B/run-e2e.sh"
python3 - "$SRC/plan-e2e.json" "$B/matrix-pinned/plan-e2e.json" "$B/bin/serve-cuda-e2e" "$FULL" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
p["serve"], p["commit"] = sys.argv[3], sys.argv[4][:8]
json.dump(p, open(sys.argv[2], "w"), indent=1)
EOF
{
  echo "Pinned $(date '+%F %T %Z') for the end-to-end agent run."
  echo "bin/serve-cuda-e2e built at $FULL, CGO_ENABLED=0 -tags cuda, from a detached worktree with its own go.work (aikit from go.mod)."
  echo "sha256 $(cat "$B/bin/serve-cuda-e2e.sha256")"
  echo "matrix-pinned/ is $HERE/matrix at $FULL; plan-e2e.json, e2e_metrics.py and run-e2e.sh from $SRC."
} > "$B/pinned.txt"
cat "$B/pinned.txt"
