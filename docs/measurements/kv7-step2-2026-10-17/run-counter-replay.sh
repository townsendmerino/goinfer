#!/usr/bin/env bash
# The server's decline counter over the night run's twelve recorded requests: the pinned serve binary, the night's flags, the bodies replayed in order at max_tokens 1.
# est 11 min (cold 18.5k-id prefill 197 s, a 9.9k-id cold one 106 s, the rest restores of ~30 s). Run under: python3 scripts/timing_lock.py run --label kv7s2-counter -- bash <this>
set -eo pipefail
export GOINFER_SWAP_GUARD=${GOINFER_SWAP_GUARD:-4096} # the night's setting (run-night.sh); the default +512 MB trips on the kernel swapping server pages (swappiness 100)
B=$HOME/goinfer-bench/kv7-step2-2026-10; N=$B/night-out; W=$B/counter-replay; mkdir -p $W; S=$W/admin.sock
$B/bin/serve-cuda-kv7s2 --model $HOME/models/mellum2.1-thinking --backend cuda --ctx 32768 --addr 127.0.0.1:18095 --log-requests --moe-cache-experts -admin-socket $S > $W/serve.log 2>&1 & SP=$!
until grep -q 'goinfer serving' $W/serve.log 2>/dev/null; do sleep 2; kill -0 $SP || { echo "server died"; tail -5 $W/serve.log; exit 1; }; done
echo "server up $(date +%T)"
python3 -I $(dirname $0)/replay-counter.py $N/bodies-kv7s2-claude-code 18095 $S $W/status.json
kill $SP; wait $SP 2>/dev/null || true
echo "done $(date +%T)"
