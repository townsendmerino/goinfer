#!/bin/bash
# P29 kernel fix: end-to-end A/B, old (origin/main kernels, ~/tmcode/goinfer-fit) vs new (~/tmcode/goinfer-p29k), order old new new old.
# Cells: Phi-3 mini q4k 512-token prefill A/B (5 pairs per run), 1.5B q4_k_m and 0.5B q4_k_m the same, then MC3 two-client on Phi-3.
set -u
OUT="$HOME/goinfer-work/p29k/out"
PHI3="$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"
Q15="$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
Q05="$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
gate() { # name need_bytes: VOID unless load1 <= 4.0 and reclaimable >= need
  l=$(sysctl -n vm.loadavg | awk '{print $2}'); m=$(cd "$HOME/tmcode/goinfer-p29k" && go run ./cmd/hostmem -reclaimable 2>&1 | tail -1)
  echo "-- cell $1 start $(date '+%T') load1 $l reclaimable $m"
  awk -v l="$l" -v m="$m" -v n="$2" 'BEGIN{exit !(l<=4.0 && m>=n)}' || { echo "VOID: $1 (load1 $l, reclaimable $m, need $2)"; return 3; }
}
echo "start $(date '+%F %T')"
gate prelude 7.7e9 || exit 3
(cd "$HOME/tmcode/goinfer-fit/backend/metal" && go test -c -tags goinfer_testhooks -o "$OUT/ab-old.test" . 2>&1 | tail -3)
(cd "$HOME/tmcode/goinfer-p29k/backend/metal" && go test -c -tags goinfer_testhooks -o "$OUT/ab-new.test" . 2>&1 | tail -3)
for arm in old new new old; do
  for cell in phi3 q15 q05; do
    case $cell in phi3) M="$PHI3"; need=7.7e9;; q15) M="$Q15"; need=4.5e9;; q05) M="$Q05"; need=3.5e9;; esac
    gate "$arm/$cell" $need || exit 3
    (cd "$HOME/tmcode/goinfer-p29k/backend/metal" && GOINFER_ZZ_AB="$M" GOMAXPROCS=4 caffeinate -i "$OUT/ab-$arm.test" -test.run '^TestZZAB$' -test.v -test.timeout 30m 2>&1 | grep -E "^\[p29-ab\]|P29 tier 1|^(--- |FAIL|PASS)|SKIP|panic")
  done
  gate "$arm/mc3-two-phi3" 8.7e9 || exit 3
  (cd "$HOME/tmcode/goinfer-p29k/backend/metal" && GOINFER_ZZ_MC3="$PHI3" GOMAXPROCS=4 caffeinate -i "$OUT/ab-$arm.test" -test.run '^TestZZMC3Two$' -test.v -test.timeout 20m 2>&1 | grep -E "^\[mc3-2\]|^(--- |FAIL|PASS)|SKIP|panic")
done
echo "end $(date '+%F %T')"
