#!/usr/bin/env bash
# S19's G-S19c (served) then G-S19d, back to back, for one timing-lock hold: the GPU stays this job's for both.
# Usage (from the repo root): run-s19-g3g4.sh <out dir> <old serve binary> <new serve binary>
set -uo pipefail
OUT=${1:?}; OLD=${2:?}; NEW=${3:?}
D=docs/measurements/multimodal-support-2026-10/s19
rc=0
bash $D/run-s19-served.sh "$OUT/g3" new="$NEW" ctl="$NEW":GOINFER_BATCHED_PREFILL=0 || rc=1
bash $D/run-s19-g4.sh "$OUT/g4" "$OLD" "$NEW" || rc=1
exit $rc
