#!/usr/bin/env bash
# Follow-up C4: C3 (2026-10-10) found C' deterministic across cold sessions and different from the resident build, and inside each C' session the first request of a prompt
# differing from its repeats (which equal each other), identically in both sessions; the resident arm is constant. C4 separates the two things that change between the first and
# the second request: the prefix reuse and the expert cache's contents. Sessions R (resident) and S (C', --moe-cache-experts), after a discarded warm request each runs A B A A
# (A = the hash-table prompt, B = the TCP prompt), non-streamed, greedy, 96 tokens, usage read for prefill_reused_tokens. Rule: docs/tasks/task-mellum21-2026-10.md, "C4",
# written before it runs. Same pinned binary as follow-up C. Correctness, not timed. Estimate ~3 min (2 loads of ~25 s, 5 requests each); queue at 10.
#   python3 scripts/night.py add mellum21-fu-c4 --est 10 --by "nobara session, Mellum2.1 follow-up C4" --doc docs/tasks/task-mellum21-2026-10.md -- bash docs/measurements/mellum21-2026-10/run-mellum21-followup-c4.sh
source "$(dirname "$0")/_common.sh"
OUT=${1:-$HOME/goinfer-logs/mellum21-followup-c4-$(date +%F)}
[ -x "$F/serve-cuda" ] || fatal "$F/serve-cuda missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"
ls "$CK".int4*.giw >/dev/null 2>&1 || fatal "the int4mix .giw cache next to $CK is missing"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
curl -fs http://127.0.0.1:18933/v1/models >/dev/null 2>&1 && fatal "something already serves on 18933"
python3 "$SRC/docs/measurements/mellum21-2026-10/cprime_determinism.py" --selftest || fatal "the classifier's self-test failed"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = R and S, requests A B A A"; exit 0; }
step drive python3 "$SRC/docs/measurements/mellum21-2026-10/cprime_determinism.py" "$F/serve-cuda" "$CK" "$OUT" --c4
grep -E "^C4|VOID" "$OUT/drive.log" | tee -a "$SUM"
grep -q 'C4 VERDICT' "$OUT/drive.log" || echo "drive: NO VERDICT LINE (a failure or VOID of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
