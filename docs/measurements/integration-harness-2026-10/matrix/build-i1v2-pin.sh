#!/bin/bash
# I1-v2's pin (docs/tasks/task-integration-harness-2026-10.md, "I1 live gate, second design (I1-v2)"): build the serve binary and
# its planted twin at one main commit REV, and copy the matrix scripts at REV beside them. Refuses a REV that does not contain
# the merge of kv3-cuda-2026-10 and the tips of connect-cc-window-2026-10 and connect-cc-model-slots-2026-10.
#
#   bash build-i1v2-pin.sh <main commit>        (by day, minutes; run before the first pilot, and not again)
#
# The planted twin differs in one place: contextOverflowError.helpless returns true, so every overflow gets the operator-facing
# wording with no overflow code. It is a measurement build and is never committed.
set -u
export PATH=$PATH:/usr/local/go/bin
REV=${1:?name the main commit to pin}
B=${B:-$HOME/goinfer-bench/i1v2-2026-10}; SRC=${SRC:-$HOME/mycode/goinfer}; WT=$B/wt-pin
REQUIRED=${REQUIRED:-kv3-cuda-2026-10 connect-cc-window-2026-10 connect-cc-model-slots-2026-10}
mkdir -p "$B/bin" || exit 2
cd "$SRC" || exit 2
REV=$(git rev-parse --verify "$REV^{commit}") || { echo "FATAL: $1 is not a commit"; exit 2; }
for br in $REQUIRED; do
  git rev-parse --verify -q "$br" >/dev/null || { echo "FATAL: branch $br does not exist here"; exit 2; }
  git merge-base --is-ancestor "$br" "$REV" || { echo "FATAL: $REV does not contain $br ($(git rev-parse --short "$br"))"; exit 2; }
done
[ -e "$B/bin/serve-cuda-i1v2" ] && { echo "FATAL: $B/bin/serve-cuda-i1v2 exists: the pin is made once"; exit 2; }
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || exit 2
sed "s#\.\./aikit/aikit#$HOME/mycode/aikit/aikit#" "$SRC/go.work" > "$WT/go.work"; [ -f "$SRC/go.work.sum" ] && cp "$SRC/go.work.sum" "$WT/go.work.sum"
(cd "$WT/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-cuda-i1v2" ./cmd/serve) || { echo "BUILD FAILED"; exit 1; }
F=$WT/internal/serveapp/overflow.go
python3 - "$F" <<'EOF' || { echo "FATAL: the planted patch did not apply"; exit 1; }
import re, sys
p = sys.argv[1]; s = open(p).read()
new, n = re.subn(r"(func \(e \*contextOverflowError\) helpless\(\) bool \{\n)\treturn [^\n]*\n", r"\1\treturn true // PLANTED (I1-v2 measurement build)\n", s)
assert n == 1, n
open(p, "w").write(new)
EOF
(cd "$WT/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$B/bin/serve-cuda-i1v2-planted" ./cmd/serve) || { echo "PLANTED BUILD FAILED"; exit 1; }
cmp -s "$B/bin/serve-cuda-i1v2" "$B/bin/serve-cuda-i1v2-planted" && { echo "FATAL: the planted binary is identical"; exit 1; }
git -C "$WT" checkout -q -- internal/serveapp/overflow.go
echo "$REV" > "$B/bin/serve-cuda-i1v2.rev"; echo "$REV planted: helpless() returns true" > "$B/bin/serve-cuda-i1v2-planted.rev"
rm -rf "$B/matrix-pinned"; mkdir -p "$B/matrix-pinned"
git archive "$REV" docs/measurements/integration-harness-2026-10/matrix | tar -x --strip-components=4 -C "$B/matrix-pinned" || exit 2
python3 - "$B/matrix-pinned/plan-i1v2.json" "$REV" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1])); p["commit"] = sys.argv[2][:10] + " (I1-v2 pin)"
open(sys.argv[1], "w").write(json.dumps(p, indent=1) + "\n")
EOF
git worktree remove --force "$WT"
go version -m "$B/bin/serve-cuda-i1v2" | grep -E 'vcs\.(revision|modified)|-tags'
echo "pinned $REV in $B"
