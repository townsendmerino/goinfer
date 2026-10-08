#!/bin/bash
# G-Q2 and G-Q3 of docs/tasks/task-metal-q4k-2026-10.md on the MacBook, night queue (the fit guard refuses Phi-3 at q4k by
# day; at night the free memory is the budget, and the guard stays ON: a refusal is recorded as such, never bypassed).
# Pinned: a detached worktree at REV with its OWN go.work (this machine's global GOWORK points at the moving main
# checkout), a test binary and a serve binary built from it. Checkpoints from ~/models.
set -u
REV=${GQ_REV:?set GQ_REV to the pinned commit}
B=$HOME/goinfer-bench/metal-q4k
OUT=$B/results-$(date +%F)
R=$HOME/tmcode/goinfer
WT=$B/wt-$REV
PHI3=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
mkdir -p "$OUT"
[ -d "$WT" ] || git -C "$R" worktree add --detach "$WT" "$REV" || exit 1
[ "$(git -C "$WT" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./metal\n)\n' > "$WT/go.work"
export GOWORK=$WT/go.work
[ -x "$B/metal-$REV.test" ] || (cd "$WT/metal" && go test -c -tags goinfer_testhooks -o "$B/metal-$REV.test" .) || exit 1
[ -x "$B/serve-metal-$REV" ] || (cd "$WT/metal" && CGO_ENABLED=0 go build -o "$B/serve-metal-$REV" ./cmd/serve) || exit 1
{ echo "rev $REV (test sha256 $(shasum -a 256 "$B/metal-$REV.test" | cut -c1-16), serve sha256 $(shasum -a 256 "$B/serve-metal-$REV" | cut -c1-16))"
  echo "phi3 $(shasum -a 256 "$PHI3" | cut -c1-16)"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1
  sysctl -n vm.swapusage; vm_stat | head -5; uptime; } | tee "$OUT/provenance.txt"
echo "=== $(date -u +%FT%TZ) START G-Q2"
# A pinned test binary runs from its package directory (it resolves ../testdata from there).
(cd "$WT/metal" && GOINFER_HEAVY_TESTS=1 "$B/metal-$REV.test" -test.run '^TestQ4KLane_realPhi3NonInferiority$' -test.v -test.count=1 -test.timeout 40m) 2>&1 | tee "$OUT/gq2.log"
rc=${PIPESTATUS[0]}
echo "=== $(date -u +%FT%TZ) END G-Q2 rc=$rc; swap $(sysctl -n vm.swapusage)"
echo "=== $(date -u +%FT%TZ) START G-Q3"
(cd "$WT" && bash docs/measurements/metal-q4k-2026-10/run-gq3-served.sh "$B/serve-metal-$REV" "$OUT/gq3" "$PHI3") 2>&1 | tee "$OUT/gq3.log"
rc2=${PIPESTATUS[0]}
echo "=== $(date -u +%FT%TZ) END G-Q3 rc=$rc2; swap $(sysctl -n vm.swapusage)"
[ $rc -eq 0 ] && rc=$rc2
exit $rc
