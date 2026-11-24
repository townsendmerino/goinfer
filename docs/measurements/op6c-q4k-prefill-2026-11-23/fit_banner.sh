#!/usr/bin/env bash
# The longest context that fits, read from the server's own plan: serve's load banner (fit:, KV plan:, the context note and the
# prefill width) for Phi-3 mini at its default load, from a serve binary built at main and one built at the branch.
# Usage: bash fit_banner.sh <main worktree> <branch worktree> <out dir>
set -uo pipefail
MAINWT=$1 BRWT=$2 OUT=$3
mkdir -p "$OUT"
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
MODEL=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
for arm in main branch; do
	wt=$MAINWT; [ $arm = branch ] && wt=$BRWT
	(cd "$wt" && git rev-parse HEAD > "$OUT/$arm.commit" && CGO_ENABLED=0 go build -C backend/cuda -tags cuda -o "$OUT/serve-$arm" ./cmd/serve) || { echo "build $arm failed"; exit 1; }
done
port=18531
for arm in main branch; do
	log=$OUT/banner-$arm.log
	"$OUT/serve-$arm" -model "$MODEL" -backend cuda -addr 127.0.0.1:$port > "$log" 2>&1 &
	pid=$!
	for _ in $(seq 240); do
		grep -q 'listening\|KV plan' "$log" && sleep 3 && break
		kill -0 $pid 2>/dev/null || break
		sleep 1
	done
	kill $pid 2>/dev/null; wait $pid 2>/dev/null
	echo "== $arm ($(cat "$OUT/$arm.commit"))"
	grep -E 'fit:|KV plan|context|prefill|ctx' "$log" | head -12
	port=$((port + 1))
done
