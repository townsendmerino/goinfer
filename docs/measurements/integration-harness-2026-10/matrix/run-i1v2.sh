#!/bin/bash
# I1-v2's runs (docs/tasks/task-integration-harness-2026-10.md, "I1 live gate, second design (I1-v2)"), unattended, from the pin
# build-i1v2-pin.sh made in $B. Holds nobara's timing lock for the whole run (a night job's lock is inherited).
#
#   PILOT=1 ONLY=i1v2-pilot-cc bash <this script>                                 (by day when the lock is free; about 30 min)
#   PILOT=1 ONLY=i1v2-pilot-codex,i1v2-pilot-opencode bash <this script>          (a night job)
#   ONLY=i1v2-codex,i1v2-codex-control,i1v2-codex-planted,i1v2-opencode,i1v2-opencode-control,i1v2-opencode-planted bash <this script>
#   ONLY=i1v2-cc,i1v2-cc-control,i1v2-cc-planted bash <this script>               (night jobs, after the pilots)
#
# A pilot that passes writes $B/pilot-ok-<agent>; a non-pilot run refuses any configuration whose agent has none.
set -u
B=${B:-$HOME/goinfer-bench/i1v2-2026-10}
ONLY=${ONLY:?name the configurations to run}
PILOT=${PILOT:-0}
DEADLINE=${DEADLINE_MINUTES:-170}
OUT=${OUT:-$B/out-$(date +%F-%H%M)-$(echo "$ONLY" | tr ',' '+' | cut -c1-40)}
PLAN=$B/matrix-pinned/plan-i1v2.json
LOCK=$HOME/mycode/goinfer/scripts/timing_lock.py
mkdir -p "$OUT" || exit 2
fatal() { echo "FATAL: $*" | tee "$OUT/FATAL.txt"; exit 2; }
for f in "$B/bin/serve-cuda-i1v2" "$B/bin/serve-cuda-i1v2.rev" "$B/matrix-pinned/matrix.py" "$B/matrix-pinned/gate_i1v2.py" "$PLAN" "$LOCK" \
         "$HOME/harnesses/node_modules/.bin/claude" "$HOME/harnesses/node_modules/.bin/codex" "$HOME/harnesses/node_modules/.bin/opencode" \
         "$HOME/harnesses/tools/bin/rg" /usr/bin/bwrap /usr/bin/socat; do
  [ -e "$f" ] || fatal "$f is missing (has build-i1v2-pin.sh been run?)"
done
case ",$ONLY," in *planted*) [ -e "$B/bin/serve-cuda-i1v2-planted" ] || fatal "the planted binary is missing";; esac
AGENTS=$(python3 - "$PLAN" "$ONLY" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1])); only = set(sys.argv[2].split(","))
names = {c["name"] for c in p["configs"]}
missing = only - names
if missing:
    sys.exit("unknown configurations: " + ",".join(sorted(missing)))
print(" ".join(sorted({a for c in p["configs"] if c["name"] in only for a in c["agents"]})))
EOF
) || fatal "the plan does not name every configuration in ONLY"
if [ "$PILOT" = 1 ]; then
  case ",$ONLY," in *,i1v2-pilot-*) ;; *) fatal "PILOT=1 runs pilot configurations only";; esac
  for c in ${ONLY//,/ }; do case $c in i1v2-pilot-*) ;; *) fatal "PILOT=1 with a non-pilot configuration: $c";; esac; done
else
  for c in ${ONLY//,/ }; do case $c in i1v2-pilot-*) fatal "a pilot configuration needs PILOT=1: $c";; esac; done
  for a in $AGENTS; do [ -e "$B/pilot-ok-$a" ] || fatal "no passing pilot for $a ($B/pilot-ok-$a)"; done
fi
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"; echo "serve: $(cat "$B/bin/serve-cuda-i1v2.rev")"
  [ -e "$B/bin/serve-cuda-i1v2-planted.rev" ] && echo "planted: $(cat "$B/bin/serve-cuda-i1v2-planted.rev")"
  echo "driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"; echo "load: $(uptime)"; free -m | sed -n 1,3p
  for h in claude codex opencode; do echo "$h: $("$HOME/harnesses/node_modules/.bin/$h" --version 2>&1 | head -1)"; done
  echo "bwrap: $(bwrap --version)"; echo "only: $ONLY"; echo "pilot: $PILOT"
} > "$OUT/provenance.txt"
cd "$B/matrix-pinned" || exit 2
python3 "$LOCK" run --label "i1v2 $ONLY" -- python3 -u matrix.py "$PLAN" "$OUT" --only "$ONLY" --deadline-minutes "$DEADLINE" 2>&1 | tee "$OUT/job.log"
rc=${PIPESTATUS[0]}
[ -s "$OUT/records.jsonl" ] || { echo "FAILED: no records were written (matrix.py exit $rc)" | tee -a "$OUT/job.log"; exit 1; }
if [ "$PILOT" = 1 ]; then
  python3 gate_i1v2.py --pilot "$B" "$OUT/records.jsonl" > "$OUT/pilot.txt" 2>&1
else
  python3 gate_i1v2.py "$OUT/records.jsonl" > "$OUT/gate-i1v2.txt" 2>&1
fi
echo "finished: $(date '+%F %T %Z') (matrix.py exit $rc)" | tee -a "$OUT/job.log"
exit "$rc"
