#!/bin/bash
# OP1 on CUDA, G1-M and G2-I on real checkpoints (docs/tasks/task-forward-architecture-2026-10.md §4.6a): every family's logits at the
# commit under test (REV) equal those at BASE (the commit before OP1's CUDA code), on this box, at int4, with the real checkpoints under
# ~/models. Runs with nobody watching: the gate binary is built here from a detached worktree at REV, and the identity tool builds
# each side's dumper from its own detached worktree.
#
#   python3 scripts/night.py add op1-identity-real-cuda --est 30 --by "<who>, OP1 CUDA (G1-M, G2-I real)" \
#     --doc docs/tasks/task-forward-architecture-2026-10.md \
#     -- env REV=<commit> BASE=78e4faf2 bash docs/measurements/op1-cuda-2026-10/run-identity-night.sh
#   DRY=1 ...   stops after the build and the provenance block.
set -uo pipefail
REV=${REV:?REV is required: the commit under test}
BASE=${BASE:?BASE is required: the commit before the CUDA OP1 code}
SRC=${SRC:-$HOME/mycode/goinfer}
B=${B:-$HOME/goinfer-bench/op1-cuda-2026-10}
OUT=${OUT:-$HOME/goinfer-logs/op1-cuda-2026-10-16/identity-$(date +%F)}
mkdir -p "$B/bin" "$OUT" || exit 2
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; using what is here"
rev=$(git rev-parse --verify "$REV^{commit}") || { echo "FATAL: $REV is not a commit here"; exit 2; }
base=$(git rev-parse --verify "$BASE^{commit}") || { echo "FATAL: $BASE is not a commit here"; exit 2; }
wt=$B/wt-gate
[ -d "$wt" ] && git worktree remove --force "$wt"
git worktree add -q --detach "$wt" "$rev" || { echo "FATAL: cannot check out $rev"; exit 2; }
(cd "$wt" && GOWORK=off go build -o "$B/bin/gate" ./cmd/gate) || { echo "FATAL: the gate build failed"; exit 2; }
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "backend: cuda"
  echo "new: $rev"; echo "base: $base"
  echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
  echo "load: $(uptime)"
} | tee "$OUT/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the build and the provenance block"; exit 0; }
"$B/bin/gate" identity "$base" "$rev" -backend cuda -assets real -timeout 20m > "$OUT/identity-real.log" 2>&1
rc=$?
echo "gate identity exit $rc (0 = every family that ran is IDENTICAL, 1 = a family DIFFERS, 2 = refused)" | tee -a "$OUT/provenance.txt"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/identity-real.log" | grep -A6 "== verdict" | tee -a "$OUT/provenance.txt"
exit $rc
