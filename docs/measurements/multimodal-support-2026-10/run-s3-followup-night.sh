#!/usr/bin/env bash
# S3's follow-ups after the 2026-10-07 night (docs/tasks/task-multimodal-support-2026-10.md, "S3 follow-ups, registered
# 2026-10-08"). Night only: each step loads Gemma 3 4B, which the fit guard refuses while the owner works. Two steps,
# independent:
#   1. Phase 2, fixed: decoder/gemma3_tower_sensitivity_real_test.go with the logits copy and serve's own prompt
#      encoding, from a test binary pinned at REV. Its reference path must reproduce G-S3b's served CPU-tower reply on the
#      same decoder (the day run's metal:cpu arm, CPU int4 with Metal's layout), or the step is void.
#   2. G-S3b's CPU-tower repeat: two metal:cpu arms on the pinned serve-metal (0c66b18b, the 2026-10-07 night's), 30 s
#      settle between arms. Each arm must decode metal-resident, or the pair is void.
# Usage: GQ_REV=<commit> run-s3-followup-night.sh [out dir]. Checkpoints from ~/models only.
set -uo pipefail
REV=${GQ_REV:?set GQ_REV to the pinned commit}
R=$HOME/tmcode/goinfer
BIN=$HOME/goinfer-bench/s3
OUT=${1:-$BIN/followup-$(date +%F)}
WT=$BIN/wt-$REV
[ -x "$BIN/serve-metal" ] && [ -f "$BIN/g3-feats.json" ] || { echo "FATAL: $BIN/serve-metal or g3-feats.json missing"; exit 2; }
[ -d "$HOME/models/gemma-3-4b-it" ] || { echo "FATAL: ~/models/gemma-3-4b-it missing"; exit 2; }
mkdir -p "$OUT"
[ -d "$WT" ] || git -C "$R" worktree add --detach "$WT" "$REV" || exit 1
[ "$(git -C "$WT" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
[ -x "$BIN/decoder-g3-$REV.test" ] || (cd "$WT/decoder" && GOWORK=off go test -c -tags realckpt -o "$BIN/decoder-g3-$REV.test" .) || exit 1
{ echo "rev $REV: decoder test sha256 $(shasum -a 256 "$BIN/decoder-g3-$REV.test" | cut -c1-16); serve-metal $(cat "$BIN/serve-metal.rev")"
  echo "started: $(date '+%F %T %Z')"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; sysctl -n vm.swapusage; uptime; } | tee "$OUT/provenance.txt"
rc=0
echo "=== 1. phase 2, fixed $(date '+%T')"
( cd "$WT/decoder" && GOINFER_HEAVY_TESTS=1 GOINFER_G3_FEATS="$BIN/g3-feats.json" \
  GOINFER_G3_SERVED_REPLY="$WT/docs/measurements/multimodal-support-2026-10/s3-gs3b/gs3c-reply-gemma-3-4b-it-metal.txt" \
  "$BIN/decoder-g3-$REV.test" -test.run '^TestGemma3TowerSensitivity$' -test.v -test.timeout 30m ) > "$OUT/1-phase2.log" 2>&1 || rc=1
grep -E "\[g3\]|^--- |VOID" "$OUT/1-phase2.log"
echo "=== 2. G-S3b CPU-tower repeat $(date '+%T')"
( cd "$WT" && GS3C_SETTLE=30 bash docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh "$BIN/serve-metal" "$OUT/2-repeat" \
  =metal:cpu,metal:cpu "$HOME/models/gemma-3-4b-it" ) > "$OUT/2-repeat.log" 2>&1 || rc=1
grep -E "decode path|IDENTICAL|differing|top-3|near-tie|exited" "$OUT/2-repeat.log"
n=$(grep -c "decode path: metal-resident" "$OUT/2-repeat.log")
[ "$n" -eq 2 ] || { echo "!! VOID: $n of 2 repeat arms decoded metal-resident"; rc=1; }
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
