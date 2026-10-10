#!/usr/bin/env bash
# Follow-up C2: is Mellum2.1's expert streaming (C') deterministic, and does it track the resident build? Follow-up C's flag: two cold C' sessions gave two different texts per
# prompt, neither equal to the resident text, though C' is documented bit-identical to resident. Five cold sessions (R1; S1a, S1b with the same history; S2 with another; S3 with
# CUDA_LAUNCH_BLOCKING=1) at ctx 2048, greedy with top-3 log-probabilities. Rule and prediction: docs/tasks/task-mellum21-2026-10.md, "C2", written before it ran.
# Same pinned binary as follow-up C (~/goinfer-bench/mellum21/followups, rev beside it). A correctness characterization, not timed; night.py still takes the timing lock.
# Estimate ~6 min (5 loads of ~25 s, 4 requests each); queue at 20.
#   python3 scripts/night.py add mellum21-fu-c2 --est 20 --by "nobara session, Mellum2.1 follow-up C2" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-followup-c2.sh
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-c2-$(date +%F)}
[ -x "$F/serve-cuda" ] || fatal "$F/serve-cuda missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"
ls "$CK".int4*.giw >/dev/null 2>&1 || fatal "the int4mix .giw cache next to $CK is missing (run a serve once, as Gate 2 did)"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
curl -fs http://127.0.0.1:18933/v1/models >/dev/null 2>&1 && fatal "something already serves on 18933"
python3 "$SRC/docs/measurements/mellum21-2026-10/cprime_determinism.py" --selftest || fatal "the classifier's self-test failed"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = R1, S1a, S1b, S2, S3 (see cprime_determinism.py)"; exit 0; }
step drive python3 "$SRC/docs/measurements/mellum21-2026-10/cprime_determinism.py" "$F/serve-cuda" "$CK" "$OUT"
grep -E "^D[1-4]|^    prompt|C2 VERDICT|VOID" "$OUT/drive.log" | tee -a "$SUM"
grep -q 'C2 VERDICT' "$OUT/drive.log" || echo "drive: NO VERDICT LINE (a failure or VOID of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
