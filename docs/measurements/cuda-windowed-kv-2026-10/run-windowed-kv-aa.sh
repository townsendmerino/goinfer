#!/usr/bin/env bash
# The A/A of docs/tasks/task-cuda-windowed-kv-2026-10.md, section 10, Mellum2.1 on nobara: the windowed-KV branch's DEFAULT path (option off) against the commit it branched from, so only the branch's own changes
# differ. Pre-registered in aa_default_path.py's docstring and the task doc. Timed (night.py holds the timing lock). Pinned: ~/goinfer-bench/windowed-kv/{serve-cuda (branch), serve-cuda-main (merge base)}, revs beside them.
# Estimate ~10 min (4 loads of ~25 s, 6 short + 3 long requests per session); queue at 20:
#   python3 scripts/night.py add windowed-kv-aa --est 20 --by "nobara session, CUDA windowed KV A/A" --doc docs/tasks/task-cuda-windowed-kv-2026-10.md -- bash <pinned src>/docs/measurements/cuda-windowed-kv-2026-10/run-windowed-kv-aa.sh
F=${F:-$HOME/goinfer-bench/windowed-kv}
source "$(dirname "$0")/../mellum21-2026-10/_common.sh"
OUT=${1:-$HOME/goinfer-logs/windowed-kv-aa-$(date +%F)}
[ -x "$F/serve-cuda" ] && [ -x "$F/serve-cuda-main" ] || fatal "$F/serve-cuda or serve-cuda-main missing"; [ -e "$CK/config.json" ] || fatal "$CK missing"
ls "$CK".int4*.giw >/dev/null 2>&1 || fatal "the int4mix .giw cache next to $CK is missing"
guard_paths; mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"; provenance
echo "main binary: rev $(cat "$F/serve-cuda-main.rev"); branch binary: rev $(cat "$F/rev")" | tee -a "$OUT/provenance.txt"
curl -fs http://127.0.0.1:18932/v1/models >/dev/null 2>&1 && fatal "something already serves on 18932"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY: preconditions hold; plan = M B B M sessions at ctx 2048, option off in both"; exit 0; }
step drive python3 "$SRC/docs/measurements/cuda-windowed-kv-2026-10/aa_default_path.py" "$F/serve-cuda-main" "$F/serve-cuda" "$CK" "$OUT"
grep -E "prompt [0-9]:|ratio|session medians|A/A|VOID" "$OUT/drive.log" | tee -a "$SUM"
grep -q 'A/A OVERALL' "$OUT/drive.log" || echo "drive: NO OVERALL LINE (a failure or VOID of this job)" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"; exit 0
