#!/bin/bash
# KS-H2 / KS-H3 readings (registered in docs/tasks/task-kernel-survey-2026-10.md before data). Run under the exclusive timing lock.
set -u
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
cd $HOME/goinfer-bench/ks-h2-h3
echo "tree $(git rev-parse --short HEAD); driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); start $(date '+%F %T %Z'); load $(cut -d' ' -f1-3 /proc/loadavg)"
apps() { echo "== card before $1:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
apps H2-R1
KSH2_MODEL=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf go test -count=1 -tags 'cuda goinfer_testhooks' ./cuda/ -run '^TestKSH2_buildVRAM$' -v -timeout 15m 2>&1 | grep -E "KS-H2|^--- |^ok|FAIL|panic"
apps H2-R2
KSH2_MODEL=$HOME/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf KSH2_CTX=16384 go test -count=1 -tags 'cuda goinfer_testhooks' ./cuda/ -run '^TestKSH2_buildVRAM$' -v -timeout 15m 2>&1 | grep -E "KS-H2|^--- |^ok|FAIL|panic"
apps H3
go test -count=1 -tags 'cuda goinfer_testhooks' ./cuda/ -run '^(TestKernelLocalMemoryCensus|TestMoERouteFirstLaunchReservation)$' -v -timeout 15m 2>&1
echo "end $(date '+%F %T %Z')"
