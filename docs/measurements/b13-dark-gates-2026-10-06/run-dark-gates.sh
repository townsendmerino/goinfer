#!/usr/bin/env bash
# The two env-gated correctness gates B13 found dark (docs/queue-engineering.md B13, 2026-10-06), as an unattended night job.
# Each is a correctness gate, not a timing run, so reading the 35B giw from the /srv/models archive is allowed (CLAUDE.md bars the archive only for
# measurements). About 2 to 6 minutes in all: the speculative gate ~35 s, the expert-paging gate 67 s with the 35B .giw in page cache and 277 s cold
# (it loads it twice). Every run streams (-v), has its own -timeout, and is logged; a SKIP is reported as a failure, because a skip is not a pass.
#
# It tests REV (default origin/main) in its own detached worktree, so whatever is half-edited in the main checkout at 3 a.m. is not what runs.
#   python3 scripts/night.py add b13-dark-gates --est 8 --by "<who>, B13" --doc docs/queue-engineering.md -- bash docs/measurements/b13-dark-gates-2026-10-06/run-dark-gates.sh
#   REV=<sha> bash docs/measurements/b13-dark-gates-2026-10-06/run-dark-gates.sh [logdir]
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
REV=${REV:-origin/main}
LOG=${1:-$HOME/goinfer-logs/b13-dark-gates-$(date +%F)}
WT=${WT:-$HOME/goinfer-bench/b13-dark-gates-wt}
export PATH=/usr/local/go/bin:$PATH
MODEL_SPEC=${MODEL_SPEC:-$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf}
MODEL_GIW=${MODEL_GIW:-/srv/models/qwen3.6-35b-a3b-int4-v12.giw}
for p in "$MODEL_SPEC" "$MODEL_GIW"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing; its gate would skip"; exit 2; }
done
mkdir -p "$LOG" "$(dirname "$WT")"
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; testing the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
cd "$WT" || exit 2
{ echo "rev:      $(git rev-parse HEAD) ($REV)"; echo "started:  $(date '+%F %T %Z')"; echo "go:       $(go version)"
  echo "spec:     GOINFER_SPEC_TARGET=$MODEL_SPEC"; echo "paging:   GOINFER_MOE_GIW=$MODEL_GIW"; } | tee "$LOG/provenance.txt"
rc=0
run() { # run <name> <timeout> <test regex> -- env...
  local name=$1 to=$2 re=$3; shift 4
  echo "== $name: start $(date '+%T')"
  env GOWORK=off "$@" go test ./decoder -count=1 -timeout "$to" -run "$re" -v 2>&1 | tee "$LOG/$name.log" | grep -E '^(--- |PASS|FAIL|ok|panic)'
  if ! grep -q -- "--- PASS: $(echo "$re" | tr -d '^$')" "$LOG/$name.log"; then echo "!! $name did not PASS (a skip or a failure; see $LOG/$name.log)"; rc=1; fi
}
run spec-target 8m '^TestSpeculativeGreedyParity_draftTarget$' -- GOINFER_SPEC_TARGET="$MODEL_SPEC"
run moe-paging 12m '^TestExpertPaging_bitExact$' -- GOINFER_MOE_GIW="$MODEL_GIW"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
