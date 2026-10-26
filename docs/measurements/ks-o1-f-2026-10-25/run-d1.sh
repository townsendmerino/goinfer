#!/bin/bash
# KS-O1-F D1 (and D3 with KSO1F_KV=f32). Runs from the ks-o1-f worktree.
set -uo pipefail
export PATH=/usr/local/go/bin:$PATH
cd ~/goinfer-bench/ks-o1-f || exit 1
echo "== $(date -Is) commit $(git rev-parse --short HEAD) KSO1F_KV=${KSO1F_KV:-}"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
uptime
GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 30m -tags 'cuda goinfer_testhooks' -run '^TestKSO1F_layerDivergence$' -v ./cuda/
echo "== exit $? $(date -Is)"
