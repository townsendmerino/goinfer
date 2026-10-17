#!/usr/bin/env bash
# Capture opencode's real request bodies for pilot task $1 (arm A0k) through a recording proxy into $2.
# Usage: capture.sh <task-number> <outdir> ; runs opencode for up to 150 s, then stops it.
set -eo pipefail
T=$1; OUT=$2; H=$HOME/goinfer-bench/restore-check-2026-10-17
mkdir -p "$OUT"
python3 -I - "$T" "$OUT" <<'PY'
import json,sys
d=json.loads(open('/home/francis/tack-runs/e1/pilot-95ed38c.jsonl').read().splitlines()[int(sys.argv[1])-1])
open(sys.argv[2]+'/statement.txt','w').write(d['statement']); open(sys.argv[2]+'/meta.txt','w').write(f"{d['repo']} {d['base']}\n")
PY
read REPO BASE < "$OUT/meta.txt"
rm -rf "$OUT/repo"; git clone -q "$HOME/tack-runs/e1/mirrors/$REPO.git" "$OUT/repo"; git -C "$OUT/repo" checkout -q "$BASE"
python3 -I "$H/proxy.py" 43298 43299 "$OUT/bodies" & PROXY=$!
sed -e 's#127.0.0.1:43213#127.0.0.1:43298#' -e "s#/home/francis/tack-runs/e1/pilot-2026-10-17-job1/task-01/A0k/repo#$OUT/repo#g" \
  "$HOME/tack-runs/e1/pilot-2026-10-17-job1/task-01/A0k/opencode.json" > "$OUT/opencode.json"
mkdir -p "$OUT/xdg"/{config,data,state}
( cd "$OUT/repo" && OPENCODE_CONFIG="$OUT/opencode.json" XDG_CONFIG_HOME="$OUT/xdg/config" XDG_DATA_HOME="$OUT/xdg/data" XDG_STATE_HOME="$OUT/xdg/state" \
  timeout 150 "$HOME/harnesses/node_modules/.bin/opencode" run --format json -m goinfer/qwen "$(cat "$OUT/statement.txt")" > "$OUT/events.jsonl" 2> "$OUT/stderr.log" ) || true
kill $PROXY 2>/dev/null || true
ls -la "$OUT/bodies"
