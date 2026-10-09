# shared by run-mellum21-followup-{a,b,c}.sh (docs/tasks/task-mellum21-2026-10.md, "Night follow-ups"). Sourced, not run.
set -uo pipefail
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
F=${F:-$HOME/goinfer-bench/mellum21/followups}      # pinned binaries: decoder.test, serve-cuda, rev
CK=${CK:-$HOME/models/mellum2.1-thinking}
GOLD=${GOLD:-$HOME/goinfer-logs/mellum21-gold}
PY=${PY:-$HOME/g4venv/bin/python}
fatal() { echo "FATAL: $*" >&2; exit 2; }
guard_paths() { case "$CK$GOLD$OUT" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac; }
provenance() { { echo "binaries: $F (rev $(cat "$F/rev" 2>/dev/null))"; echo "checkpoint: $CK; goldens: $GOLD"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2p; df -h "$HOME" | tail -1; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader; } | tee "$OUT/provenance.txt"; }
step() { local name=$1; shift; local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/$name.log" | cut -c1-120)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-12s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
gt() { local run=$1; shift; ( cd "$SRC/decoder" && env GOINFER_HEAVY_TESTS=1 GOINFER_MELLUM_CKPT="$CK" "$@" "$F/decoder.test" -test.run "^${run}\$" -test.v -test.timeout 170m ); }
verdict() { local name=$1 test=$2
  if grep -q -- "--- PASS: $test " "$OUT/$name.log" && ! grep -q -- "--- SKIP: $test " "$OUT/$name.log"; then echo "$name: PASS" | tee -a "$SUM"
  elif grep -q -- "--- SKIP: $test " "$OUT/$name.log"; then echo "$name: SKIPPED (a failure of this job)" | tee -a "$SUM"
  else echo "$name: FAIL (see $OUT/$name.log)" | tee -a "$SUM"; fi; }
