#!/usr/bin/env bash
# Re-run the two env-gated correctness gates B13 found dark (docs/queue-engineering.md B13, 2026-10-06). Each is a correctness gate, not a timing run,
# so reading the 35B giw from the /srv/models archive is allowed (CLAUDE.md bars the archive only for measurements). About 2 to 6 minutes in all:
# the speculative gate ~35 s, the expert-paging gate 67 s with the 35B .giw in page cache, 277 s cold (it loads it twice). Every run streams (-v), has its own -timeout, and is logged.
#   bash docs/measurements/b13-dark-gates-2026-10-06/run-dark-gates.sh [logdir]
set -uo pipefail
cd "$(dirname "$0")/../../.."
LOG=${1:-$HOME/goinfer-logs/b13-dark-gates-$(date +%F)}
mkdir -p "$LOG"
rc=0
run() { # run <name> <timeout> <test regex> -- env...
  local name=$1 to=$2 re=$3; shift 4
  echo "== $name: start $(date '+%T')"
  if env "$@" go test ./decoder -count=1 -timeout "$to" -run "$re" -v 2>&1 | tee "$LOG/$name.log" | grep -E '^(--- |PASS|FAIL|ok)'; then :; fi
  if ! grep -q -- "--- PASS: $(echo "$re" | tr -d '^$')" "$LOG/$name.log"; then echo "!! $name did not PASS (a skip is not a pass)"; rc=1; fi
}
run spec-target 8m '^TestSpeculativeGreedyParity_draftTarget$' -- GOINFER_SPEC_TARGET="$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
run moe-paging 12m '^TestExpertPaging_bitExact$' -- GOINFER_MOE_GIW=/srv/models/qwen3.6-35b-a3b-int4-v12.giw
echo "logs: $LOG; rc=$rc"
exit $rc
