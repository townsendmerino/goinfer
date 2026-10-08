#!/usr/bin/env bash
# S16 on the Mac, night (docs/tasks/task-multimodal-support-2026-10.md, "S16's night job", as amended 2026-10-08 before it
# runs): Qwen2.5-VL-3B (which the fit guard refuses by day) and Qwen3-VL-2B. Pinned: detached worktrees with their own
# go.work (this machine's global GOWORK points at the moving main checkout): the test binary from S16_REV (main, with the
# re-registered bar's text control), the serve binary from S16_SERVE_REV (the local branch s16-night-on: main with both
# production switches on; main keeps them off until these steps pass). Three independent steps.
#   1. G-S16c, real: TestS16MRoPEPrefill_real (both models, the four F2a images, the re-registered bar: non-inferiority
#      to a same-length text control, min minus 0.005, argmax differences only at near-ties).
#   2. G-S16c, served, per model: one image request (run-gs3c-served.sh's: table.png, 32 greedy tokens, top-3
#      log-probabilities) through the serve binary with --exact-prefill (today's path: the CPU prefill and the upload) and
#      without (S16), each beside a CPU reference arm. PASS: the two Metal replies identical, or their first difference at
#      a near-tie in today's path's log-probabilities (p(other) >= half p(top)).
#   3. Speed, a record, per model: image-turn TTFT, today's path against S16, vision_ttft.py, 3 rounds of 1 warm-up + 3
#      timed, the order rotated, a new image every request (gemma3_preprocess_image.png, S7's 896² image).
# Usage: S16_REV=<main commit> S16_SERVE_REV=<s16-night-on commit> run-s16-night.sh [out dir]. Checkpoints from ~/models only.
set -uo pipefail
REV=${S16_REV:?set S16_REV to the pinned main commit}
SREV=${S16_SERVE_REV:?set S16_SERVE_REV to the s16-night-on commit}
R=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/s16
OUT=${1:-$B/night-$(date +%F)}
MODELS="qwen25vl-3b-instruct qwen3-vl-2b-instruct"
for m in $MODELS; do [ -d "$HOME/models/$m" ] || { echo "FATAL: ~/models/$m is missing"; exit 2; }; done
mkdir -p "$OUT"
wt() { # wt <rev>: a detached worktree at rev with its own go.work; prints its path
  local w=$B/wt-$1
  [ -d "$w" ] || git -C "$R" worktree add --detach "$w" "$1" >&2 || return 1
  [ "$(git -C "$w" rev-parse --short=8 HEAD)" = "$1" ] || { echo "worktree is not at $1" >&2; return 1; }
  printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./metal\n)\n' > "$w/go.work"
  echo "$w"
}
WT=$(wt "$REV") || exit 1
SWT=$(wt "$SREV") || exit 1
[ -x "$B/metal-$REV.test" ] || (cd "$WT/metal" && GOWORK=$WT/go.work go test -c -tags goinfer_testhooks -o "$B/metal-$REV.test" .) || exit 1
[ -x "$B/serve-metal-$SREV" ] || (cd "$SWT/metal" && GOWORK=$SWT/go.work CGO_ENABLED=0 go build -o "$B/serve-metal-$SREV" ./cmd/serve) || exit 1
SERVE=$B/serve-metal-$SREV
{ echo "test rev $REV (sha256 $(shasum -a 256 "$B/metal-$REV.test" | cut -c1-16)), serve rev $SREV (sha256 $(shasum -a 256 "$SERVE" | cut -c1-16))"
  echo "started: $(date '+%F %T %Z')"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; sysctl -n vm.swapusage; uptime; } | tee "$OUT/provenance.txt"
rc=0
echo "=== 1. G-S16c real $(date '+%T')"
(cd "$WT/metal" && GOINFER_HEAVY_TESTS=1 "$B/metal-$REV.test" -test.run '^TestS16MRoPEPrefill_real$' -test.v -test.count=1 -test.timeout 90m) > "$OUT/1-real.log" 2>&1 || rc=1
grep -E "S16 G-S16c|^--- |^    ---" "$OUT/1-real.log"
SERVED=docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh
for mname in $MODELS; do
  M=$HOME/models/$mname
  echo "=== 2. G-S16c served, $mname $(date '+%T')"
  (cd "$WT" && GS3C_SETTLE=30 GS3C_EXTRA="--exact-prefill" bash "$SERVED" "$SERVE" "$OUT/2-old-$mname" =cpu,metal "$M") > "$OUT/2-old-$mname.log" 2>&1 || rc=1
  (cd "$WT" && GS3C_SETTLE=30 bash "$SERVED" "$SERVE" "$OUT/2-new-$mname" =cpu,metal "$M") > "$OUT/2-new-$mname.log" 2>&1 || rc=1
  grep -E "decode path|IDENTICAL|differing|near-tie" "$OUT/2-old-$mname.log" "$OUT/2-new-$mname.log"
  python3 "$WT/docs/measurements/multimodal-support-2026-10/s16/served_verdict.py" "$OUT" "$mname" | tee "$OUT/2-verdict-$mname.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  echo "=== 3. speed, $mname $(date '+%T')"
  python3 - "$OUT/plan-$mname.json" "$SERVE" "$M" "$WT/testdata/gemma3_preprocess_image.png" "$mname" <<'PY'
import json, sys
out, serve, m, img, name = sys.argv[1:]
base = {"cell": name, "media": "image", "base": img, "prompt": "Describe this image.", "api": "openai", "model": ""}
cells = [dict(base, engine="today", port=18630, cmd=[serve, "--model", m, "--backend", "metal", "--exact-prefill", "--addr", "127.0.0.1:18630"]),
         dict(base, engine="s16", port=18631, cmd=[serve, "--model", m, "--backend", "metal", "--addr", "127.0.0.1:18631"])]
json.dump({"rounds": 3, "per_round": 3, "rotate": True, "cells": cells}, open(out, "w"), indent=1)
PY
  (cd "$WT" && python3 docs/measurements/multimodal-support-2026-10/vision_ttft.py "$OUT/plan-$mname.json" "$OUT/3-speed-$mname") > "$OUT/3-speed-$mname.log" 2>&1 || rc=1
  tail -8 "$OUT/3-speed-$mname.log"
done
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
