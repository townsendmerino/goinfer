#!/usr/bin/env bash
# Runs toolcall_smoke.py on three arms, one after another, each under the shared timing lock:
# the candidate on Metal (the default backend on a Mac), the candidate on the CPU, and the previous release on the CPU.
#   bash run-smoke.sh                                         # runs 1 and 2: the replay, keywords, sampled and auto groups
#   SMOKE_GROUPS=required,auto5 TAG=-run3 bash run-smoke.sh   # run 3: the two groups where the model picks the tool
# TAG is appended to each output's name, so a later run does not overwrite an earlier one's results.
set -uo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:/usr/local/bin:$PATH
D=$HOME/goinfer-work/eval-followup/toolsmoke
N=$HOME/goinfer-bench/toolsmoke-new-wt
M=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
SMOKE_GROUPS=${SMOKE_GROUPS:-replay,keywords,sampled,auto}
TAG=${TAG:-}
[ -f "$M" ] || { echo "FATAL: no model at $M"; exit 2; }
rc=0
for arm in "metal serve-metal-bfce89962 metal 8197" "cpu-new serve-cpu-bfce89962 cpu 8198" "cpu-old serve-cpu-v0.25.0 cpu 8199"; do
  set -- $arm
  echo "== arm $1: $2 -backend $3, groups $SMOKE_GROUPS, started $(date '+%H:%M:%S %Z'), load $(uptime | sed 's/.*load averages: //')"
  python3 "$N/scripts/timing_lock.py" run --shared --wait 3000 --label "tool-call smoke $1" -- \
    python3 "$D/toolcall_smoke.py" "$D/$2" "$3" "$M" "$D/result-$1$TAG.json" --port "$4" --groups "$SMOKE_GROUPS" \
    --transcript "$N/scripts/w4_transcript_base.json" 2>&1 | tee "$D/smoke-$1$TAG.log"
  r=${PIPESTATUS[0]}; echo "arm $1 rc=$r, finished $(date '+%H:%M:%S %Z')"; [ "$r" -ne 0 ] && rc=1
done
echo "all arms done rc=$rc"
exit $rc
