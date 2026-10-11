#!/usr/bin/env bash
# Follow-up C3: C2's registered fallback. C2 (2026-10-10) found C' deterministic and history-independent at its short history and did not reproduce follow-up C's finding (two
# cold C' sessions, two different texts per prompt). C3 repeats follow-up C's OWN request shape: after a discarded warm request, each of the three prompts three times in a row,
# streamed, greedy, 128 tokens, thinking off; sessions R1 (resident), S1a and S1b (C', --moe-cache-experts, identical). Rule: docs/tasks/task-mellum21-2026-10.md, "C3", written before it runs.
# Same pinned binary as follow-up C. Correctness, not timed; night.py still takes the timing lock. Estimate ~4 min (3 loads of ~25 s, 10 requests each); queue at 15.
#   python3 scripts/night.py add mellum21-fu-c3 --est 15 --by "nobara session, Mellum2.1 follow-up C3" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-followup-c3.sh
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-c3-$(date +%F)}
[ -x "$F/serve-cuda" ] || fatal "$F/serve-cuda missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"
ls "$CK".int4*.giw >/dev/null 2>&1 || fatal "the int4mix .giw cache next to $CK is missing"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
curl -fs http://127.0.0.1:18933/v1/models >/dev/null 2>&1 && fatal "something already serves on 18933"
python3 "$SRC/docs/measurements/mellum21-2026-10/cprime_determinism.py" --selftest || fatal "the classifier's self-test failed"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = R1, S1a, S1b at follow-up C's history, streamed"; exit 0; }
step drive python3 "$SRC/docs/measurements/mellum21-2026-10/cprime_determinism.py" "$F/serve-cuda" "$CK" "$OUT" --c-history
grep -E "^C3|VOID" "$OUT/drive.log" | tee -a "$SUM"
grep -q 'C3 VERDICT' "$OUT/drive.log" || echo "drive: NO VERDICT LINE (a failure or VOID of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
