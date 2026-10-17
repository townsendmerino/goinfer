#!/usr/bin/env bash
# Capture opencode's request for task-05's failing turn (session copy cut before the garbage reply), then replay it restored x2 and cold x2.
set -eo pipefail
H=$HOME/goinfer-bench/restore-check-2026-10-17; cd $H; T=$H/cap-t05
SYN="The previous request exceeded the provider's size limit due to large media attachments. The conversation was compacted and media files were removed from context. If the user was asking about attached images or files, explain that the attachments were too large to process and suggest they try again with smaller or fewer files.

Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
echo "=== $(date +%T) capture"
./serve.sh $H/serve-t05cap.log & SP=$!
until grep -q 'goinfer serving' $H/serve-t05cap.log 2>/dev/null; do sleep 2; done
python3 -I proxy.py 43298 43299 $T/bodies & PROXY=$!
sed -e 's#127.0.0.1:43213#127.0.0.1:43298#' $T/opencode.json > $T/opencode.run.json
( cd $T/repo && OPENCODE_CONFIG=$T/opencode.run.json XDG_CONFIG_HOME=$T/xdg/config XDG_DATA_HOME=$T/xdg/data XDG_STATE_HOME=$T/xdg/state \
  timeout 120 $HOME/harnesses/node_modules/.bin/opencode run --session ses_eb59209feffeSq2UjZdDxENI2d --format json -m goinfer/qwen "$SYN" > $T/events.jsonl 2> $T/stderr.log ) || true
kill $PROXY $SP 2>/dev/null; wait $SP 2>/dev/null || true
ls -la $T/bodies
