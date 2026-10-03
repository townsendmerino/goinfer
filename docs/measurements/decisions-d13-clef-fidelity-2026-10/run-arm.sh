#!/bin/bash
# D13 fidelity: ONE arm of goinfer's Clef pipeline over the D10 records (docs/measurements/decisions-d13-clef-fidelity-2026-10-03.md, pre-registered before this ran).
# Unattended: a PRE-BUILT test binary at a registered sha256 (the tree may move), the records and encoder dump sha256-checked, the model from ~/models (never the archive),
# resumable (rows already written are skipped), nothing that needs a Claude session. The reference rows must exist: this job grades NOTHING, it writes the arm's rows; the
# morning grader (grade.py) needs the reference's probs_f32.jsonl from the d10-clef-f32 job, and this script refuses to run without it so a failed D10 job does not burn the night.
#   ARM=int4 bash run-arm.sh            all 150 records
#   ARM=f32 EVERY=3 bash run-arm.sh     every 3rd record (the registered stride for that arm), when the pre-registration says so
# Fixture-quality generation, not a timed measurement: no latency figure from this is quoted as a speed (D14 owns speed); the throughput it logs is for planning only.
# Exits non-zero, loudly, on every refusal. (A silent exit under `set -e` killed both D10 jobs on 2026-10-02: no command here can fail inside $(...) | ... .)
set -uo pipefail
B=${B:-$HOME/goinfer-bench/decisions-d13}
REPO=${REPO:-$HOME/mycode/goinfer}
ARM=${ARM:?int4, int8int8 or f32}
EVERY=${EVERY:-1}
BIN=$B/clef-fidelity.test; BIN_SHA=${BIN_SHA:?the registered sha256 of $BIN}
MODEL=$HOME/models/clef-flash
REF=${REF:-$HOME/goinfer-bench/decisions-d10/out/probs_f32.jsonl}
OUT=$B/out/probs_goinfer_$ARM.jsonl
die() { echo "$(date +%T) REFUSED: $*"; exit 1; }
sum() { sha256sum "$1" | cut -d' ' -f1; }
case "$MODEL" in /srv/models/*|/Volumes/*) die "model $MODEL is on the archive, not the bench set";; esac
[ -s "$MODEL/joint_head.safetensors" ] || die "no local Clef-flash at $MODEL (rsync it from /srv/models/clef-flash first)"
[ -x "$BIN" ] || die "no binary $BIN"
[ "$(sum "$BIN")" = "$BIN_SHA" ] || die "binary sha256 differs from the registered one"
[ "$(sum "$REPO/testdata/decisions/clef/records.jsonl")" = 5942ccb997012d69c37945ef8cf3a2dd1974f62009ef14d9a668f5d4c49f9b7c ] || die "records.jsonl differs from the registered one"
[ -s "$REF" ] || die "no reference rows at $REF (the d10-clef-f32 job did not finish): nothing to grade against"
nref=$(wc -l < "$REF"); [ "$nref" -eq 150 ] || die "reference has $nref rows, want 150"
mkdir -p "$B/out"
n0=0; [ ! -f "$OUT" ] || n0=$(wc -l < "$OUT")
echo "$(date +%T) == D13 arm $ARM start (every $EVERY); rows already written: $n0; load $(cut -d' ' -f1-3 /proc/loadavg); free $(free -g | awk '/Mem/{print $7}') GB"
cd "$REPO/internal/clef" || die "no $REPO/internal/clef"
CLEF_FIDELITY_ARM="$ARM" CLEF_FIDELITY_OUT="$OUT" CLEF_FIDELITY_EVERY="$EVERY" CLEF_MODEL_DIR="$MODEL" CLEF_FIDELITY_WALL=${WALL:-170} \
  "$BIN" -test.run 'TestFidelityArm_run$' -test.v -test.timeout "${TIMEOUT:-3h}"
rc=$?
n=0; [ ! -f "$OUT" ] || n=$(wc -l < "$OUT")
want=$(( (150 + EVERY - 1) / EVERY ))
echo "$(date +%T) == test binary rc=$rc; rows $n of $want"
[ "$rc" -eq 0 ] || die "the test binary failed (rc $rc)"
[ "$n" -ge "$want" ] || die "only $n of $want rows written"
echo "$(date +%T) == DONE arm $ARM; rows in $OUT"
