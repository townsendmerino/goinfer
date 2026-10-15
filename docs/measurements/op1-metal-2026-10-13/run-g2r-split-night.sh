#!/bin/bash
# OP1's real Gemma 2 gate, G2-R, as a night job on the MacBook, in two processes (docs/tasks/task-forward-architecture-2026-10.md
# §4.6, "G2-R amendment, 2026-10-15"): TestG2R_cpuReference loads only CPU models and writes the CPU side of every arm to
# a file; TestG2R_registered then loads only Metal models and grades against it. Same arms, prompts, bars and voids as
# run-g2r-night.sh; each process's loads meet the fit guard and Metal's memory guard alone. No guard is bypassed.
# Amendment 2: the CPU process loads CPU .giw bundles of the same .gguf files; any that are missing are built here first, at
# REV, by cmd/prequant (about 10.6 GB for the five; the build refuses under 14 GB free).
# Amendment 3: the Metal process loads Metal .giw bundles. The int8int8 ones are hard links to the CPU bundles (the same
# bytes); the int4 ones (the 2B, and the 0.5B for the Metal precondition) are built with -target metal (about 2.7 GB with
# the 0.5B's int8int8; refused under 6 GB free). A refused load is VOID, not FAIL. Run TestG2R_fitCheck by day first.
#
#   python3 scripts/night.py add op1-g2r-metal --est 30 --by "<who>, OP1 Metal (G2-R, split)" --doc docs/tasks/task-forward-architecture-2026-10.md \
#     -- env REV=<declaration branch commit> bash docs/measurements/op1-metal-2026-10-13/run-g2r-split-night.sh
#
# Exit: 0 when G2-R PASSES; 1 when it FAILS; 2 when it is VOID or SKIPPED (a skip is not a pass), the CPU process did not
# write its reference, or the setup failed. Estimate: the bundles on a first run (five prequant builds, about 10 minutes),
# a cold build, the CPU process (about 18 CPU loads and the sha256 reads), then the Metal process (about 11 Metal loads):
# about 40 minutes on a first run, 25 after; queue at 45.
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
{ echo "started: $(date '+%F %T %Z')"; echo "rev: $rev"; echo "load: $(uptime)"; echo "disk free: $(df -h "$HOME" | tail -1 | awk '{print $4}')"; } | tee "$OUT/provenance.txt"
REF=$OUT/g2r-cpu-ref.gob
rm -f "$REF"
GIW=$B/giw
mkdir -p "$GIW"
G2=$HOME/models/gemma-2-2b-it-Q8_0.gguf
DENSE=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
SMALL=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
[ -e "$SMALL" ] || { echo "FATAL: $SMALL is missing"; exit 2; }
missing=()
for spec in "$G2:int8int8" "$G2:int4" "$LINK:int8int8" "$DENSE:int8int8" "$DENSE:int4"; do
  src=${spec%:*}; q=${spec##*:}; dst=$GIW/$(basename "${src%.gguf}").$q.cpu.giw
  [ -s "$dst" ] || missing+=("$src:$q:$dst")
done
if [ ${#missing[@]} -gt 0 ]; then
  free_kb=$(df -k "$HOME" | tail -1 | awk '{print $4}')
  if [ "$free_kb" -lt $((14 * 1024 * 1024)) ]; then
    echo "VOID: ${#missing[@]} CPU bundle(s) to build and only $((free_kb / 1024 / 1024)) GB free (14 needed)" | tee -a "$OUT/provenance.txt"
    [ "$made_link" = 1 ] && rm -f "$LINK"; cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null; exit 2
  fi
  for m in "${missing[@]}"; do
    src=${m%%:*}; rest=${m#*:}; q=${rest%%:*}; dst=${rest#*:}
    echo "== building $(basename "$dst"): $(date '+%T')" | tee -a "$OUT/provenance.txt"
    (cd "$WT" && go run ./cmd/prequant -quant "$q" -o "$dst.tmp" "$src" && mv "$dst.tmp" "$dst" && { [ ! -e "$dst.tmp.verified" ] || mv "$dst.tmp.verified" "$dst.verified"; }) 2>&1 | tail -3 | tee -a "$OUT/provenance.txt"
    [ -s "$dst" ] || { echo "VOID: building $dst failed" | tee -a "$OUT/provenance.txt"; [ "$made_link" = 1 ] && rm -f "$LINK"; cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null; exit 2; }
  done
fi
# Amendment 3's Metal bundles: int8int8 as hard links to the CPU bundles, int4 built for the Metal target.
metal_bundles() {
  local spec src q cpu dst
  for spec in "$G2:int8int8" "$LINK:int8int8" "$DENSE:int8int8" "$SMALL:int8int8" "$G2:int4" "$SMALL:int4"; do
    src=${spec%:*}; q=${spec##*:}; dst=$GIW/$(basename "${src%.gguf}").$q.metal.giw
    [ -s "$dst" ] && continue
    free_kb=$(df -k "$HOME" | tail -1 | awk '{print $4}')
    [ "$free_kb" -ge $((6 * 1024 * 1024)) ] || { echo "VOID: building $(basename "$dst") with only $((free_kb / 1024 / 1024)) GB free (6 needed)"; return 1; }
    if [ "$q" = int8int8 ]; then
      cpu=$GIW/$(basename "${src%.gguf}").$q.cpu.giw
      if [ ! -s "$cpu" ]; then
        echo "== building $(basename "$cpu"): $(date '+%T')"
        (cd "$WT" && go run ./cmd/prequant -quant "$q" -o "$cpu.tmp" "$src" && mv "$cpu.tmp" "$cpu" && { [ ! -e "$cpu.tmp.verified" ] || mv "$cpu.tmp.verified" "$cpu.verified"; }) 2>&1 | tail -3
      fi
      ln "$cpu" "$dst" || { echo "VOID: linking $(basename "$dst") failed"; return 1; }
    else
      echo "== building $(basename "$dst"): $(date '+%T')"
      (cd "$WT" && go run ./cmd/prequant -target metal -quant "$q" -o "$dst.tmp" "$src" && mv "$dst.tmp" "$dst" && { [ ! -e "$dst.tmp.verified" ] || mv "$dst.tmp.verified" "$dst.verified"; }) 2>&1 | tail -3
    fi
    [ -s "$dst" ] || { echo "VOID: $(basename "$dst") is missing after its build"; return 1; }
  done
}
if ! metal_bundles 2>&1 | tee -a "$OUT/provenance.txt" || grep -q '^VOID' "$OUT/provenance.txt"; then
  [ "$made_link" = 1 ] && rm -f "$LINK"; cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null; exit 2
fi
run() { # <test> <log>
  (cd "$WT" && env GOINFER_HEAVY_TESTS=1 GOINFER_G2R_REF="$REF" GOINFER_G2R_COMMIT="$rev" GOINFER_G2R_GIW_DIR="$GIW" caffeinate -i \
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
# Amendment 3: FAIL is G2-R's own rule only (arm G failing, or parked), read from the test's verdict lines. A registered
# stop is reported as STOPPED. Anything else that ends the run (a refused load, a precondition, a void rule, a crash, a
# skip) measured nothing and is VOID.
if grep -q '^--- PASS: TestG2R_registered' "$OUT/g2r.log"; then rc=0; v=PASS
elif grep -q 'VERDICT: VOID' "$OUT/g2r.log"; then rc=2; v="VOID as a softcap test (registered)"
elif grep -q 'VERDICT: G2-R FAILS' "$OUT/g2r.log"; then rc=1; v=FAIL
elif grep -q 'VERDICT: G2-R PARKED' "$OUT/g2r.log"; then rc=1; v="PARKED (band)"
elif grep -q 'STOP' "$OUT/g2r.log"; then rc=1; v="STOPPED (a registered stop; see the log)"
else rc=2; v="VOID (refused, a precondition, a void rule, a crash, or a skip; see the logs)"; fi
echo "finished: $(date '+%F %T %Z') verdict $v rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
