#!/bin/bash
# OP6's Metal tier 1 (docs/tasks/task-forward-architecture-2026-10.md §4.7) on the MacBook, night queue: Phi-3 mini at
# q4k does not fit by day. Two steps, the load-time fit guard ON for both (no bypass; a refusal is the reading):
#   1. the fit guard's q4k price against Phi-3's measured weight bytes (PR #41's unmeasured row);
#   2. TestQ4KStep_phi3Tier1: the step route bit for bit against the sequential loop, the golden's argmax, the speed A/B.
# Pinned: a detached worktree at OP6_REV with its own go.work, test binaries built from it. Checkpoint from ~/models.
# Each step's peak RSS comes from /usr/bin/time -l on the test binary itself.
set -u
REV=${OP6_REV:?set OP6_REV to the pinned commit}
B=$HOME/goinfer-bench/op6-q4k-step
OUT=$B/results-$(date +%F)
R=$HOME/tmcode/goinfer
WT=$B/wt-$REV
PHI3=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
[ -e "$PHI3" ] || { echo "FATAL: $PHI3 is missing"; exit 2; }
mkdir -p "$OUT"
[ -d "$WT" ] || git -C "$R" worktree add --detach "$WT" "$REV" || exit 1
[ "$(git -C "$WT" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./metal\n)\n' > "$WT/go.work"
export GOWORK=$WT/go.work
[ -x "$B/decoder-$REV.test" ] || (cd "$WT" && go test -c -o "$B/decoder-$REV.test" ./decoder) || exit 1
[ -x "$B/metal-$REV.test" ] || (cd "$WT/metal" && go test -c -o "$B/metal-$REV.test" .) || exit 1
{ echo "rev $REV (decoder test sha256 $(shasum -a 256 "$B/decoder-$REV.test" | cut -c1-16), metal test sha256 $(shasum -a 256 "$B/metal-$REV.test" | cut -c1-16))"
  echo "phi3 $(shasum -a 256 "$PHI3" | cut -c1-16)"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1
  sysctl -n vm.swapusage; vm_stat | head -5; uptime; } | tee "$OUT/provenance.txt"

# step <name> <dir> <binary> <run pattern> <PASS line>: a skip is not a pass, so the step ran only if its PASS line printed.
step() {
  echo "=== $(date -u +%FT%TZ) START $1; swap $(sysctl -n vm.swapusage)"
  (cd "$2" && GOINFER_HEAVY_TESTS=1 /usr/bin/time -l "$3" -test.run "$4" -test.v -test.count=1 -test.timeout 15m) > "$OUT/$1.log" 2>&1
  local rc=$?
  cat "$OUT/$1.log"
  if [ $rc -eq 0 ] && ! grep -q "^$5" "$OUT/$1.log"; then
    echo "$1 DID NOT RUN: no '$5' line (a skip exits 0)"; rc=5
  fi
  echo "=== $(date -u +%FT%TZ) END $1 rc=$rc; swap $(sysctl -n vm.swapusage)"
  return $rc
}
step price "$WT/decoder" "$B/decoder-$REV.test" '^TestFitEstimate_q4kAgreesWithResidentWeightBytes$/Phi-3' \
  '    --- PASS: TestFitEstimate_q4kAgreesWithResidentWeightBytes/Phi-3' || exit $?
step tier1 "$WT/metal" "$B/metal-$REV.test" '^TestQ4KStep_phi3Tier1$' '--- PASS: TestQ4KStep_phi3Tier1'
