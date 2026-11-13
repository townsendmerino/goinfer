#!/bin/bash
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
cd ~/wt/v1-2026-11-12 || exit 1
python3 scripts/timing_lock.py run --label v1-gate-identity-cuda -- /home/francis/goinfer-logs/v1-2026-11-12/bin/gate identity -backend cuda -keep -logdir /home/francis/goinfer-logs/v1-2026-11-12/identity-run 51ed2ae27 4edbe93e9
echo "=== EXIT $? $(date +%T)"
