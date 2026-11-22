#!/usr/bin/env bash
# One rerun of the parity sweep's decoder cell narrowed to TestProgram_L_everyParameterIsLive, at 36dbfbaf1 in the sweep's own worktree:
# the 2026-11-21 sweep's decoder cell hit its 180m package timeout with only this test unfinished (36m27s in; 1463 s at v0.24.0).
# Same tool and timeout; waits for the sweep and the KV10 discriminator, then takes nobara's exclusive timing lock.
set -uo pipefail
D=$HOME/goinfer-logs/relgate-2026-11-21/progl-rerun
NS=$HOME/goinfer-bench/night-src
WT=$HOME/goinfer-bench/relgate-parity/wt
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH CGO_ENABLED=0
mkdir -p "$D/records"
for p in 5318 15474; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
echo "sweep and kv10 done at $(date '+%F %T')"
[ "$(git -C "$WT" rev-parse HEAD)" = 36dbfbaf149aabdff43dce5766b49a65183113a7 ] || { echo "FATAL: worktree not at 36dbfbaf1"; exit 2; }
cd "$WT" || exit 2
exec python3 "$NS/scripts/timing_lock.py" run --wait 7200 --label "parity rerun TestProgram_L 36dbfbaf1 (coordinator)" -- \
  env GOWORK=off REALCKPT=0 GATE_RUN='^TestProgram_L_everyParameterIsLive$' go run ./cmd/gate parity -logdir "$D" -record "$D/records"
