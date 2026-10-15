#!/bin/bash
# OP1's real Gemma 2 gate, G2-R, as a night job on the MacBook, in two processes (docs/tasks/task-forward-architecture-2026-10.md
# §4.6, "G2-R amendment, 2026-10-15"): TestG2R_cpuReference loads only CPU models and writes the CPU side of every arm to
# a file; TestG2R_registered then loads only Metal models and grades against it. Same arms, prompts, bars and voids as
# run-g2r-night.sh; each process's loads meet the fit guard and Metal's memory guard alone. No guard is bypassed.
#
#   python3 scripts/night.py add op1-g2r-metal --est 30 --by "<who>, OP1 Metal (G2-R, split)" --doc docs/tasks/task-forward-architecture-2026-10.md \
#     -- env REV=<declaration branch commit> bash docs/measurements/op1-metal-2026-10-13/run-g2r-split-night.sh
#
# Exit: 0 when G2-R PASSES; 1 when it FAILS; 2 when it is VOID or SKIPPED (a skip is not a pass), the CPU process did not
# write its reference, or the setup failed. Estimate: a cold build, the CPU process (about 14 CPU loads and 3 sha256 reads
# of 6.4 GB), then the Metal process (about 11 Metal loads): about 25 minutes; queue at 30.
set -uo pipefail
REV=${REV:?REV is required: the declaration branch commit with TestG2R_cpuReference}
SRC=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/op1-metal-2026-10
WT=$B/wt-g2r
OUT=$HOME/goinfer-logs/op1-metal-2026-10-13/g2r-split-night-$(date +%F)
LINK=$HOME/models/gemma-3-4b-it-Q4_K_M.gguf
G3=$HOME/models/gemma-3-4b-it-gguf/gemma-3-4b-it-Q4_K_M.gguf
command -v go >/dev/null 2>&1 || export PATH=/usr/local/go/bin:/opt/homebrew/bin:$PATH
[ "$(uname -s)" = Darwin ] || { echo "FATAL: the Metal half's script"; exit 2; }
for f in "$HOME/models/gemma-2-2b-it-Q8_0.gguf" "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$G3"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
mkdir -p "$B" "$OUT" || exit 2
cd "$SRC" || exit 2
rev=$(git rev-parse --verify "$REV^{commit}") || { echo "FATAL: $REV is not a commit here"; exit 2; }
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$rev" || { echo "FATAL: cannot check out $rev"; exit 2; }
(cd "$WT" && go work init . ./gpu ./metal) || { echo "FATAL: go work init"; exit 2; }
# The Gemma 3 control reads the checkpoint at the top of ~/models; link it there for this session only (registered).
made_link=0
if [ ! -e "$LINK" ]; then ln -s "$G3" "$LINK" && made_link=1; fi
REF=$OUT/g2r-cpu-ref.gob
rm -f "$REF"
{ echo "started: $(date '+%F %T %Z')"; echo "rev: $rev"; echo "load: $(uptime)"; echo "disk free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"; } | tee "$OUT/provenance.txt"
run() { # <test> <log>
  (cd "$WT" && env GOINFER_HEAVY_TESTS=1 GOINFER_G2R_REF="$REF" GOINFER_G2R_COMMIT="$rev" caffeinate -i \
    go test -tags goinfer_testhooks ./metal -run "^$1\$" -count=1 -v -timeout 60m) 2>&1 | tee "$OUT/$2"
}
echo "== process 1, the CPU reference: $(date '+%T')" | tee -a "$OUT/provenance.txt"
run TestG2R_cpuReference g2r-cpu.log
if grep -q '^--- PASS: TestG2R_cpuReference' "$OUT/g2r-cpu.log" && [ -s "$REF" ]; then
  echo "== process 2, the Metal arms: $(date '+%T')" | tee -a "$OUT/provenance.txt"
  run TestG2R_registered g2r.log
else
  echo "VOID: the CPU process did not write its reference (see g2r-cpu.log); the Metal arms did not run" | tee -a "$OUT/provenance.txt"
  : >"$OUT/g2r.log"
fi
[ "$made_link" = 1 ] && rm -f "$LINK"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
if grep -q '^--- PASS: TestG2R_registered' "$OUT/g2r.log"; then rc=0; v=PASS
elif grep -q '^--- FAIL: TestG2R_registered' "$OUT/g2r.log"; then rc=1; v=FAIL
else rc=2; v="VOID (skipped, refused, or did not finish)"; fi
echo "finished: $(date '+%F %T %Z') verdict $v rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
