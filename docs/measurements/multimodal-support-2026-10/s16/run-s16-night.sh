#!/usr/bin/env bash
# S16 on the Mac, night (docs/tasks/task-multimodal-support-2026-10.md, "S16, the night job, registered 2026-10-08"):
# Qwen2.5-VL-3B, which the fit guard refuses by day. Pinned: a detached worktree at S16_REV with its own go.work (this
# machine's global GOWORK points at the moving main checkout), a test binary and a serve binary built from it. Three
# independent steps; one failing does not stop the others.
#   1. G-S16c, real: TestS16MRoPEPrefill_real (the four F2a images, the resident m-RoPE prefill against the upload bridge,
#      both decoding on Metal; cosine >= 0.9999 and every argmax equal over the last row and 8 steps).
#   2. G-S16c, served: one image request (run-gs3c-served.sh's: table.png, 32 greedy tokens, top-3 log-probabilities)
#      through the same binary with --exact-prefill (today's path: the CPU prefill and the upload) and without (S16),
#      each beside a CPU reference arm. PASS: the two Metal replies identical, or their first difference at a near-tie in
#      today's path's log-probabilities (p(other) >= half p(top)).
#   3. Speed, a record: image-turn TTFT, today's path against S16, vision_ttft.py, 3 rounds of 1 warm-up + 3 timed, the
#      order rotated, a new image every request (gemma3_preprocess_image.png, S7's 896² image), serve's defaults otherwise.
# Usage: S16_REV=<commit> run-s16-night.sh [out dir]. Checkpoints from ~/models only.
set -uo pipefail
REV=${S16_REV:?set S16_REV to the pinned commit}
R=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/s16
OUT=${1:-$B/night-$(date +%F)}
WT=$B/wt-$REV
M=$HOME/models/qwen25vl-3b-instruct
[ -d "$M" ] || { echo "FATAL: $M is missing"; exit 2; }
mkdir -p "$OUT"
[ -d "$WT" ] || git -C "$R" worktree add --detach "$WT" "$REV" || exit 1
[ "$(git -C "$WT" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./metal\n)\n' > "$WT/go.work"
export GOWORK=$WT/go.work
[ -x "$B/metal-$REV.test" ] || (cd "$WT/metal" && go test -c -tags goinfer_testhooks -o "$B/metal-$REV.test" .) || exit 1
[ -x "$B/serve-metal-$REV" ] || (cd "$WT/metal" && CGO_ENABLED=0 go build -o "$B/serve-metal-$REV" ./cmd/serve) || exit 1
SERVE=$B/serve-metal-$REV
{ echo "rev $REV (test sha256 $(shasum -a 256 "$B/metal-$REV.test" | cut -c1-16), serve sha256 $(shasum -a 256 "$SERVE" | cut -c1-16))"
  echo "started: $(date '+%F %T %Z')"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1; sysctl -n vm.swapusage; uptime; } | tee "$OUT/provenance.txt"
rc=0
echo "=== 1. G-S16c real $(date '+%T')"
(cd "$WT/metal" && GOINFER_HEAVY_TESTS=1 "$B/metal-$REV.test" -test.run '^TestS16MRoPEPrefill_real$' -test.v -test.count=1 -test.timeout 40m) > "$OUT/1-real.log" 2>&1 || rc=1
grep -E "S16 G-S16c|^--- |^    ---" "$OUT/1-real.log"
echo "=== 2. G-S16c served $(date '+%T')"
SERVED=docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh
(cd "$WT" && GS3C_SETTLE=30 GS3C_EXTRA="--exact-prefill" bash "$SERVED" "$SERVE" "$OUT/2-old" =cpu,metal "$M") > "$OUT/2-old.log" 2>&1 || rc=1
(cd "$WT" && GS3C_SETTLE=30 bash "$SERVED" "$SERVE" "$OUT/2-new" =cpu,metal "$M") > "$OUT/2-new.log" 2>&1 || rc=1
grep -E "decode path|IDENTICAL|differing|near-tie" "$OUT/2-old.log" "$OUT/2-new.log"
python3 - "$OUT" <<'EOF' | tee "$OUT/2-verdict.txt"
import json, math, sys, os
out = sys.argv[1]
fam = "qwen25vl-3b-instruct"
def load(d):
    return (open(f"{out}/{d}/gs3c-reply-{fam}-metal.txt").read(), json.load(open(f"{out}/{d}/gs3c-logprobs-{fam}-metal.json")))
try:
    (ro, lo), (rn, ln) = load("2-old"), load("2-new")
except Exception as e:
    print(f"G-S16c served: VOID ({e})"); sys.exit(1)
if ro == rn:
    print("G-S16c served: PASS, IDENTICAL replies (today's path against S16)"); sys.exit(0)
i = 0
while i < min(len(lo), len(ln)) and lo[i]["token"] == ln[i]["token"]:
    i += 1
if i >= min(len(lo), len(ln)):
    print("G-S16c served: FAIL, the replies differ in length only"); sys.exit(1)
top = lo[i]["top_logprobs"]; ptop = math.exp(top[0]["logprob"])
po = [math.exp(c["logprob"]) for c in top if c["token"] == ln[i]["token"]]
near = bool(po and po[0] >= ptop / 2)
print(f"first differing generated token {i}: today's {lo[i]['token']!r}, S16 {ln[i]['token']!r}; today's top-3: " +
      ", ".join(f"{c['token']!r} {math.exp(c['logprob']):.3f}" for c in top))
print(f"G-S16c served: {'PASS' if near else 'FAIL'} (near-tie: p(other) >= {ptop/2:.3f}: {near})"); sys.exit(0 if near else 1)
EOF
[ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
echo "=== 3. speed $(date '+%T')"
python3 - "$OUT/plan.json" "$SERVE" "$M" "$WT/testdata/gemma3_preprocess_image.png" <<'PY'
import json, sys
out, serve, m, img = sys.argv[1:]
base = {"cell": "qwen2.5-vl-3b", "media": "image", "base": img, "prompt": "Describe this image.", "api": "openai", "model": ""}
cells = [dict(base, engine="today", port=18630, cmd=[serve, "--model", m, "--backend", "metal", "--exact-prefill", "--addr", "127.0.0.1:18630"]),
         dict(base, engine="s16", port=18631, cmd=[serve, "--model", m, "--backend", "metal", "--addr", "127.0.0.1:18631"])]
json.dump({"rounds": 3, "per_round": 3, "rotate": True, "cells": cells}, open(out, "w"), indent=1)
PY
(cd "$WT" && python3 docs/measurements/multimodal-support-2026-10/vision_ttft.py "$OUT/plan.json" "$OUT/3-speed") > "$OUT/3-speed.log" 2>&1 || rc=1
tail -8 "$OUT/3-speed.log"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
