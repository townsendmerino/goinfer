#!/usr/bin/env bash
# OP1 on CUDA, the device gate's failed parts re-run (owner, 2026-10-16). The first run (gate gpu at b04b0273) failed in three places that were not OP1's:
#   - check 2a, a build error in cuda/kv2_optionoff_test.go (fixed by #66, which also makes CI vet the cuda tests with -tags cuda alone);
#   - the heavy tier's gemma4-26b and qwen3.6-35b groups, and the two fresh-process tests TestP20ExpertLocality and TestSpecPagerInteraction, all on
#     v16 sidecars the v17 build refuses. The two sidecars were rebuilt at v17 (cmd/prequant -quant int4 -target canonical, from the sources the
#     v16 rebuild record names), and the two fresh-process tests passed by day on them.
# This runs the same four things the gate runs, in the gate's own forms, on REV in a detached worktree. Everything else the gate checks passed in
# the first run (PTX 48/48, graphs, drain group, parity, the other heavy groups, hygiene), and this script does not repeat it. The device gate is
# recorded green only if every part here passes and the first run's other parts stand. A skip counts as a failure.
#   python3 scripts/night.py add op1-gate-rerun-cuda --est 75 --by "<who>, OP1 CUDA (device gate re-run)" --doc docs/tasks/task-forward-architecture-2026-10.md \
#     -- env REV=<main commit with #66> bash docs/measurements/op1-cuda-2026-10/run-gate-rerun-night.sh
#   DRY=1 ...   stops after provisioning and the sidecar check.
# Estimate: 2a a few minutes; the 26B group 14 min and the 35B group 7 min (B23, docs/measurements/b23-cuda-heavy-split-2026-10-15.md); the two fresh
# tests 1 + 3 min by day; provisioning and cold reads of the 14.6 and 20 GB sidecars. About 45 min; queue at 75.
set -uo pipefail
SRC=${SRC:-$HOME/mycode/goinfer}
REV=${REV:?REV is required: the commit under test (main, with #66)}
LOG=${1:-$HOME/goinfer-logs/gate-rerun-night-$(date +%F)}
WT=${WT:-$HOME/goinfer-bench/gate-rerun-night-wt}
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$LOG" "$(dirname "$WT")" || exit 2
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; testing the local $REV"
rev=$(git rev-parse --verify "$REV^{commit}") || { echo "FATAL: $REV is not a commit here"; exit 2; }
# the two sidecars must be v17 files, or the parts below would fail for the old reason (inner header at 0x40: GINFW, then the version)
for n in gemma4-26b-int4.giw qwen3.6-35b-a3b-int4.giw; do
  v=$(python3 - "$HOME/models/$n" <<'PY'
import struct, sys
try:
    b = open(sys.argv[1], "rb").read(0x49)
    print(struct.unpack("<I", b[0x45:0x49])[0] if b[0x40:0x45] == b"GINFW" else "badmagic")
except OSError as e:
    print("missing")
PY
)
  [ "$v" = 17 ] || { echo "FATAL: $n is $v, not a v17 file: rebuild it (cmd/prequant -quant int4 -target canonical)"; exit 2; }
done
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$rev" || { echo "FATAL: cannot check out $rev"; exit 2; }
EXC="$LOG/untracked-fixtures.exclude"
fixtures=$(bash "$SRC/scripts/worktree_fixtures.sh" "$SRC" "$WT" "$EXC") || { echo "FATAL: placing the fixtures: $fixtures"; exit 2; }
cd "$WT" || exit 2
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.excludesFile GIT_CONFIG_VALUE_0="$EXC"
go work init . ./gpu ./cuda ./metal ./demo/agent || { echo "FATAL: go work init"; exit 2; }
{ echo "rev: $(git rev-parse HEAD)"; echo "started: $(date '+%F %T %Z')"; echo "fixtures: $fixtures"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(uptime)"; } | tee "$LOG/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after provisioning and the sidecar check"; exit 0; }
rc=0; part() { # part <name> <required PASS lines, space separated> <log> <command...>: run, then require rc 0, no FAIL and every named test passed
  local name=$1 want=$2 out=$3; shift 3
  echo "== $name: started $(date '+%T')"
  "$@" > "$out" 2>&1; local r=$?
  local bad=0; [ $r -ne 0 ] && bad=1; grep -q -- "--- FAIL" "$out" && bad=1
  for t in $want; do grep -q -- "--- PASS: $t " "$out" || { echo "   no '--- PASS: $t' in $out"; bad=1; }; done
  if [ $bad = 0 ]; then echo "   PASS $name"; else echo "   FAIL $name (rc=$r): $out"; rc=1; fi
}
part "2a kernel-level suite (-tags cuda alone, -short)" "" "$LOG/2a.log" env CGO_ENABLED=0 go test -count=1 -timeout 40m -short -tags cuda -v ./cuda/
part "fresh-process TestP20ExpertLocality" "TestP20ExpertLocality" "$LOG/iso-p20.log" env CGO_ENABLED=0 GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestP20ExpertLocality$' -v ./cuda/
part "fresh-process TestSpecPagerInteraction" "TestSpecPagerInteraction" "$LOG/iso-specpager.log" env CGO_ENABLED=0 GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestSpecPagerInteraction$' -v ./cuda/
for g in gemma4-26b qwen3.6-35b; do
  echo "== heavy group $g: started $(date '+%T')"
  go run ./cmd/gate cuda-heavy -group "$g" -logdir "$LOG" > "$LOG/heavy-$g.log" 2>&1; r=$?
  clean=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG/heavy-$g.log")
  if [ $r -eq 0 ] && ! grep -q -- "--- FAIL\|FAIL " <<<"$clean"; then echo "   PASS group $g"; else echo "   FAIL group $g (rc=$r): $LOG/heavy-$g.log"; rc=1; fi
  echo "   skips in group $g: $(grep -c -- "--- SKIP" <<<"$clean")"
done
echo "RESULT: $([ $rc = 0 ] && echo 'ALL PARTS PASS' || echo 'A PART FAILED') (rc=$rc); skips are listed per group above and are read by hand: a skip is not a pass" | tee -a "$LOG/provenance.txt"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
