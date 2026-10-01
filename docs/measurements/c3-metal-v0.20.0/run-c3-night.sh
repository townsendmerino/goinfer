#!/usr/bin/env bash
# C3 (RELEASING.md: "Metal consumer window", owed because v0.20.0 bumps aikit v1.45.1 -> v1.51.1), the two parts that
# need an idle Mac. Parts 1, 3 and 5 ran by day on 2026-10-01 (docs/measurements/c3-metal-consumer-window-v0.20.0.md).
# By day the fit guard refused even the 1.5B (3.8 GB available with the owner's apps open, a 2.6 GB budget against
# 4.7 GB priced), and a bypass on this 16 GB Mac is a standing no.
#
# Both run at the published metal/v0.20.0 commit (8e4fb57c), not the root tag's commit.
#   Part 2, decode tok/s against the 73.6 claim: TestZZ_metalDepthBench (resident ForwardArgmax, decode only,
#     min of 5 batches, W4A8, qwen2.5-coder-1.5b-instruct-q4_k_m) with GOWORK=off, so goinfer resolves at v0.20.0
#     from the module proxy, as a consumer's would. Reported against 73.6 and v0.18.0's 71.6 (same harness);
#     depth 128 below 70.0 (more than 5% under the claim) is a finding to investigate.
#   Part 4, the Metal device gate (§C1-M): the release's own run-metal-gate.sh, copied here unchanged, at 8e4fb57c.
#     Pass: the gate's verdict, all 9 declared groups reporting and zero FAIL.
#
#   python3 scripts/night.py add c3-metal-v020 --est 25 ... -- bash ~/goinfer-bench/c3-v0.20.0/run-c3-night.sh
set -euo pipefail
REV=8e4fb57c
SRC=$HOME/tmcode/goinfer
BASE=$HOME/goinfer-bench/c3-v0.20.0
WT=$BASE/wt-night
LOG=$HOME/goinfer-logs/c3-v0.20.0
mkdir -p "$BASE" "$LOG"

cd "$SRC"
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree prune
git worktree add -q --detach "$WT" "$REV"
{
  echo "rev:      $(git -C "$WT" rev-parse HEAD) (metal/v0.20.0 = $(git rev-parse metal/v0.20.0))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
} | tee "$LOG/provenance.txt"

echo "== part 2: depth bench — $(date '+%T')" | tee -a "$LOG/provenance.txt"
(cd "$WT/metal" && GOINFER_METAL_DEPTH_BENCH=1 GOWORK=off go test -count=1 -run '^TestZZ_metalDepthBench$' -v -timeout 30m .) \
  2>&1 | tee "$LOG/depth-bench.log" || true
git worktree remove --force "$WT"

echo "== part 4: device gate — $(date '+%T')" | tee -a "$LOG/provenance.txt"
REV=$REV bash "$BASE/run-metal-gate.sh" 2>&1 | tee "$LOG/device-gate.log" || true
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
