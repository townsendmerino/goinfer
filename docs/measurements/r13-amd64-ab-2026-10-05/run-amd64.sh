#!/usr/bin/env bash
# R-13, the amd64 arm (pre-registered in ../r13-amd64-ab-2026-10-05.md, committed before this runs): TestR13_cpuAB on nobara, in-process, one process per model set.
# A pre-built test binary (built from 3db00301, which includes 9863cb4c: the fused gate+up honours w4a8PreOff on amd64; the first build, from 575a3768, was discarded unrun. The tree may move before tonight). Nothing here needs a Claude session alive.
D=$HOME/goinfer-logs/r13
BIN=$D/decoder.test
if [ ! -x "$BIN" ]; then echo "FATAL: $BIN is missing or not executable"; exit 2; fi
for m in "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/models/qwen15-moe-a27b-l4slice"; do
  [ -e "$m" ] || { echo "FATAL: $m is missing"; exit 2; }
  case "$m" in /srv/models/*) echo "FATAL: $m is under /srv/models, not a bench surface"; exit 2;; esac
done
idle() { for i in $(seq 1 90); do l=$(cut -d" " -f1 /proc/loadavg); awk -v l="$l" "BEGIN{exit !(l<1.0)}" && return 0; echo "=== waiting for idle: loadavg $l"; sleep 20; done; return 1; }
idle || { echo "FATAL: box never went idle"; exit 3; }
cd $HOME/mycode/goinfer/decoder || exit 2
echo "=== $(date -u +%FT%TZ) START TestR13_cpuAB (1.5B and the Qwen1.5-MoE slice)"
GOINFER_R13_AB=1 "$BIN" -test.run '^TestR13_cpuAB$' -test.v -test.count=1 -test.timeout 40m > "$D/r13-amd64-ab.log" 2>&1
rc=$?
echo "=== $(date -u +%FT%TZ) END rc=$rc"
grep -E "RESULT|FAIL|differs" "$D/r13-amd64-ab.log" | cut -c1-300
if ! grep -q "RESULT" "$D/r13-amd64-ab.log"; then echo "FATAL: no RESULT line in $D/r13-amd64-ab.log (the harness refused or died)"; exit 4; fi
exit $rc
