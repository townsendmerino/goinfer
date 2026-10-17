#!/bin/bash
# Pin KV7 step 2's night rerun (README.md) by day: build serve at REV (the branch's head) from a detached worktree and copy the matrix harness at REV, this
# directory's plan and runner, the e2e extractor and the repo's snapshot-prefix test beside it, so the night job runs what was pinned. Needs no GPU.
#
#   REV=<commit> B=$HOME/goinfer-bench/kv7-step2-2026-10 bash pin-kv7s2.sh
set -euo pipefail
REV=${REV:?the commit to pin}
B=${B:?the pinned directory}
REPO=${REPO:-$HOME/mycode/goinfer}
HERE=docs/measurements/integration-harness-2026-10
SRC=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B/bin"
WT="$B/wt"
[ -d "$WT" ] || git -C "$REPO" worktree add --detach "$WT" "$REV" >/dev/null
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
(cd "$WT" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-cuda-kv7s2" github.com/townsendmerino/goinfer/cuda/cmd/serve)
echo "$FULL" > "$B/bin/serve-cuda-kv7s2.rev"
sha256sum "$B/bin/serve-cuda-kv7s2" | cut -d' ' -f1 > "$B/bin/serve-cuda-kv7s2.sha256"
rm -rf "$B/matrix-pinned"
cp -r "$WT/$HERE/matrix" "$B/matrix-pinned"
cp "$WT/docs/measurements/e2e-agents-2026-10-16/e2e_metrics.py" "$B/matrix-pinned/"
cp "$SRC/run-night.sh" "$B/run-night.sh"
python3 - "$SRC/plan-kv7s2.json" "$B/matrix-pinned/plan-kv7s2.json" "$B/bin/serve-cuda-kv7s2" "$FULL" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
p["serve"], p["commit"] = sys.argv[3], sys.argv[4][:8]
json.dump(p, open(sys.argv[2], "w"), indent=1)
EOF
{
  echo "Pinned $(date '+%F %T %Z') for KV7 step 2's night rerun."
  echo "bin/serve-cuda-kv7s2 built at $FULL, CGO_ENABLED=0 -tags cuda, from a detached worktree with its own go.work."
  echo "sha256 $(cat "$B/bin/serve-cuda-kv7s2.sha256")"
  echo "wt/ is the repo at $FULL (the snapshot-prefix test runs from it); matrix-pinned/ is $HERE/matrix at $FULL."
} > "$B/pinned.txt"
cat "$B/pinned.txt"
