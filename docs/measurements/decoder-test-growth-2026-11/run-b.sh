#!/usr/bin/env bash
# decgrowth B (Francis, 2026-11-22 ~12:05 PST, "b yes one run"): quick -full's decoder cell, alone, at b208993e8 (origin/main, includes #1218), under nobara's exclusive
# timing lock. The command is the one cmd/gate builds for the cell (cell-<rev>/decoder.args.json, printed by a scratch cmd/gate test that reproduces the v0.25.0 120m
# command exactly at 36dbfbaf1), with -count=1, the 120m timeout and GOINFER_MEM_PROBE=1 (a test hook: docs/env-vars.md). GOWORK=off, as quick runs the root module.
set -uo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
L=$HOME/goinfer-logs/decgrowth-2026-11-22; WT=$HOME/goinfer-bench/decgrowth-wt; REV=b208993e8282a540ca7142e8c3b3ab6c367c0bc5
cd "$WT" || exit 2
[ "$(git rev-parse HEAD)" = "$REV" ] || { echo "FATAL: not $REV"; exit 2; }
[ "$(git status --porcelain | wc -l)" = 0 ] || { echo "FATAL: dirty tree"; git status --porcelain | head; exit 2; }
mapfile -t ARGS < <(python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); a.insert(2,"-count=1"); print("\n".join(a))' "$L/cell-$REV/decoder.args.json")
printf '%q ' "${ARGS[@]}" > "$L/decoder-cmd.txt"
echo "== start $(date '+%F %T %Z') rev $REV; go $(go version); fixtures: $(cat "$L/fixtures.txt" | head -c 200)" | tee "$L/run-b.log"
python3 "$HOME/mycode/goinfer/scripts/timing_lock.py" run --wait 1800 --label "decgrowth B: quick -full decoder cell alone at b208993e8 (Francis 2026-11-22), est 80 min, ends by 21:30" -- \
  env GOWORK=off GOINFER_MEM_PROBE=1 "${ARGS[@]}" > "$L/decoder.json" 2> "$L/decoder.stderr"
rc=$?
echo "== end $(date '+%F %T %Z') rc=$rc" | tee -a "$L/run-b.log"
gzip -kf "$L/decoder.json"
