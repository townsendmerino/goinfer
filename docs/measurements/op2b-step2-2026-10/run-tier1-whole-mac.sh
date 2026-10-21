#!/bin/bash
# OP2b's tier 1 on the whole NVIDIA-Nemotron-Nano-9B-v2 on Metal (docs/tasks/task-forward-architecture-2026-10.md §4.7),
# night queue: B27's v17 int4 bundle is 7.27 GB, which this Mac holds only with the desktop's memory free. The fit guard
# and Metal's memory guard stay on; a refused load is VOID (rc 2), never bypassed.
# Pinned: a detached worktree at B27_REV with its own go.work and a metal test binary built from it. Bundles from ~/models.
set -u
REV=${B27_REV:?set B27_REV to the pinned commit}
B=$HOME/goinfer-bench/op2b-tier1-whole
OUT=$B/results-$(date +%F)
R=$HOME/tmcode/goinfer
WT=$B/wt-$REV
NEMO=$HOME/models/nemotron/nemotron-nano-9b-v2-int4-metal.v17.giw
CTRL=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw
for f in "$NEMO" "$CTRL"; do [ -e "$f" ] || { echo "VOID: $f is missing"; exit 2; }; done
mkdir -p "$OUT"
[ -d "$WT" ] || git -C "$R" worktree add --detach "$WT" "$REV" || exit 1
[ "$(git -C "$WT" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./metal\n)\n' > "$WT/go.work"
export GOWORK=$WT/go.work
[ -x "$B/metal-$REV.test" ] || (cd "$WT/metal" && go test -c -tags goinfer_testhooks -o "$B/metal-$REV.test" .) || exit 1
{ echo "rev $REV (metal test sha256 $(shasum -a 256 "$B/metal-$REV.test" | cut -c1-16))"
  echo "nemotron $(cut -c1-16 "$NEMO.sha256" 2>/dev/null)"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1
  sysctl -n vm.swapusage; vm_stat | head -5; uptime; } | tee "$OUT/provenance.txt"
echo "=== $(date -u +%FT%TZ) START tier1-whole; swap $(sysctl -n vm.swapusage)"
(cd "$WT/metal" && GOINFER_HEAVY_TESTS=1 /usr/bin/time -l "$B/metal-$REV.test" -test.run '^TestNemotronResident_tier1Whole$' \
  -test.v -test.count=1 -test.timeout 20m) > "$OUT/tier1-whole.log" 2>&1
rc=$?
cat "$OUT/tier1-whole.log"
echo "=== $(date -u +%FT%TZ) END tier1-whole rc=$rc; swap $(sysctl -n vm.swapusage)"
grep -q 'VOID\|REFUSED\|needs ~' "$OUT/tier1-whole.log" && { echo "VOID: a load was refused or an input missing"; exit 2; }
if [ $rc -eq 0 ] && ! grep -q '^--- PASS: TestNemotronResident_tier1Whole' "$OUT/tier1-whole.log"; then
  echo "DID NOT RUN: no PASS line (a skip exits 0)"; exit 5
fi
exit $rc
