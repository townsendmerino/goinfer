#!/bin/bash
# D14: Clef-flash against JEV-9B, speed (docs/measurements/decisions-d14-clef-speed-2026-10-03.md, pre-registered before this ran). Written for the MAC's night queue
# (macOS: bash 3.2, no sha256sum, no setsid) and equally runnable on Linux. Unattended: a PINNED serve binary at a registered sha256, the frozen D7 prompts sha256-checked by the
# harness, both models from ~/models (never the archive or the SMB mount), a fresh output directory, every refusal an echo and a non-zero exit.
#   SERVE=<pinned serve> SERVE_SHA=<sha256> D14_STATES=6 D14_KS=256,1024 bash run-d14.sh
set -uo pipefail
B=${B:-$HOME/goinfer-bench/decisions-d14}
SERVE=${SERVE:?the pinned goinfer-serve binary}
SERVE_SHA=${SERVE_SHA:?its registered sha256}
JEV=${D14_JEV:-$HOME/models/JEV-9B}
CLEF=${D14_CLEF:-$HOME/models/clef-flash}
STATES=${D14_STATES:-6}
KS=${D14_KS:-256,1024}
HERE=$(cd "$(dirname "$0")" && pwd)
die() { echo "$(date +%T) REFUSED: $*"; exit 1; }
sum() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
for d in "$JEV" "$CLEF"; do
  case "$d" in /srv/models/*|/Volumes/*) die "model $d is on the archive or the SMB mount, not the bench set";; esac
done
[ -s "$JEV/head.safetensors" ] || die "no JEV-9B head at $JEV (head.safetensors)"
[ -s "$CLEF/joint_head.safetensors" ] || die "no Clef-flash at $CLEF (joint_head.safetensors)"
[ -x "$SERVE" ] || die "no binary $SERVE"
[ "$(sum "$SERVE")" = "$SERVE_SHA" ] || die "serve binary sha256 differs from the registered one"
OUT="$B/run-$(date +%F)"
RAW="$OUT/raw.jsonl"
mkdir -p "$OUT" || die "cannot create $OUT"
[ ! -e "$RAW" ] || die "$RAW exists: a re-run must not mix rows"
echo "$(date +%T) == D14 start; host $(uname -srm); serve $(basename "$SERVE"); states $STATES; ks $KS; load $(uptime | sed 's/.*load average[s]*: //')"
GOINFER_SERVE_BIN="$SERVE" D14_JEV="$JEV" D14_CLEF="$CLEF" python3 -u "$HERE/d14_bench.py" run --raw "$RAW" --log "$OUT/servers.log" --states "$STATES" --ks "$KS"
rc=$?
n=0; [ ! -f "$RAW" ] || n=$(grep -c '"arm"' "$RAW")
nks=$(echo "$KS" | awk -F, '{print NF}')
want=$(( 8 * nks * STATES ))
echo "$(date +%T) == run rc=$rc; rows $n of $want"
python3 "$HERE/d14_bench.py" analyze --raw "$RAW" | tee "$OUT/analysis.txt"
[ "$rc" -eq 0 ] || die "the harness failed (rc $rc)"
[ "$n" -eq "$want" ] || die "expected $want rows, got $n"
echo "$(date +%T) == DONE; results in $OUT"
