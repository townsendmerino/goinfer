#!/bin/bash
# The Mac's night run for Part V of docs/tasks/task-giw-v16-transcode-2026-10.md:
#   G-V1's real Metal cells: logits identity, v15 against v16, each build through its own transcode, on the 1.5B and the 7B.
#   G-V3: the Metal alias banner's "bound in place" figure on the 7B under each format.
# Pre-built binaries in ~/goinfer-bench/v16-manual/bin (BUILD.txt beside them names the revisions). Sidecars are written
# in that scratch directory, never in ~/models. A cell that does not run is a failure here, not a pass. Estimate: 15 min.
set -uo pipefail
B=$HOME/goinfer-bench/v16-manual
D=$HOME/goinfer-logs/giw-v16/$(date +%Y-%m-%d)
REPO=$HOME/tmcode/goinfer
BASE=c07c7f5d # main before Part V: weights format v15
NEW=d8f1918a # Part V: weights format v16
mkdir -p "$D" "$B/m7"
rc=0

echo "== G-V1, real Metal cells (started $(date '+%H:%M:%S'))"
cd "$REPO" || exit 2
"$B/bin/gate" identity -giw logits -assets real -backend metal -only coder-1.5b-q4km,7b-q4km "$BASE" "$NEW" > "$D/g-v1-real-metal.log" 2>&1
grc=$?
sed -n '/== verdict ==/,$p' "$D/g-v1-real-metal.log"
[ $grc -eq 0 ] || { echo "G-V1: the gate exited $grc"; rc=1; }
# The report lists every other family as NOT RUN (only two assets are asked for), so each asked-for cell is checked by
# its own line: IDENTICAL, and on the Metal resident path. A cell Metal declined has no such line.
for a in coder-1.5b-q4km 7b-q4km; do
  grep -q "IDENTICAL  $a @ int4.*metal-resident" "$D/g-v1-real-metal.log" || { echo "G-V1: no IDENTICAL metal-resident line for $a; a skip is not a pass"; rc=1; }
done

echo "== G-V3, the 7B's alias banner under each format (started $(date '+%H:%M:%S'))"
cd "$B" || exit 2
ln -sf "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" m7/q7.gguf
find m7 -name '*.giw*' -delete
./serve-once.sh bin/serve-v15 m7/q7.gguf "$D/g-v3-7b-v15.log" 18441 q7 > "$D/g-v3-7b-v15.reply" 2>&1
./serve-once.sh bin/serve-v16 m7/q7.gguf "$D/g-v3-7b-v16.log" 18442 q7 > "$D/g-v3-7b-v16.reply" 2>&1
find m7 -name '*.giw*' -delete
for v in v15 v16; do
  if ! grep 'weights aliased' "$D/g-v3-7b-$v.log"; then
    echo "G-V3: the $v build printed no alias banner (Metal did not build the 7B resident):"; grep 'decode path' "$D/g-v3-7b-$v.log"; rc=1
  fi
done
grep -q 'does not load' "$D/g-v3-7b-v16.log" || { echo "G-V3: the v16 build did not refuse and rebuild the v15 sidecar"; rc=1; }
echo "records in $D; exit $rc (finished $(date '+%H:%M:%S'))"
exit $rc
