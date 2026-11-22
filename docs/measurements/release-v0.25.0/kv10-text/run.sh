#!/usr/bin/env bash
# KV10 discriminator (coordinator, 2026-11-21): is the depth-65536 FAIL at 36dbfbaf1 caused by the test's text (built from
# the repo's own files) or by the code? One test binary built at 36dbfbaf1, run twice under nobara's exclusive timing
# lock: (A) cwd in a 36dbfbaf1 tree (reproduces today's text), (B) cwd in an e0d661712 tree (yesterday's text).
set -uo pipefail
D=$HOME/goinfer-logs/relgate-2026-11-21/kv10-text
NS=$HOME/goinfer-bench/night-src
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH CGO_ENABLED=0
G=$(cat "$HOME/goinfer-logs/relgate-2026-11-21/gate-gpu/pid")
PP=${PARITY_PID:-}
while kill -0 "$G" 2>/dev/null; do sleep 30; done
if [ -n "$PP" ]; then while kill -0 "$PP" 2>/dev/null; do sleep 30; done; fi
echo "gate and parity launcher done at $(date '+%F %T')"
for r in 36dbfbaf149aabdff43dce5766b49a65183113a7 e0d661712; do
  w=$HOME/goinfer-bench/kv10-text-${r:0:9}
  [ -d "$w" ] || git -C "$NS" worktree add -q --detach "$w" "$r"
  (cd "$w" && bash scripts/make_work.sh >/dev/null 2>&1)
done
cd "$HOME/goinfer-bench/kv10-text-36dbfbaf1" || exit 2
go test -c -tags 'cuda goinfer_testhooks' -o "$D/kv10.test" ./backend/cuda || { echo "FATAL: build"; exit 2; }
exec python3 "$NS/scripts/timing_lock.py" run --wait 7200 --label "kv10 text discriminator (coordinator)" -- bash "$D/both.sh"
