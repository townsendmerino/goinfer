#!/bin/bash
# VK1 K1/K3 under Intel SDE, v2: the go toolchain never runs under SDE. K3's parent runs under
# sde64 -no-follow-child; its coverage child is a natively built -cover test binary run under its own sde64 -<chip>.
export PATH=/usr/local/go/bin:$PATH
W=/home/francis/goinfer-bench/vk1-sde-wt
L=/home/francis/goinfer-bench/vk1-sde-logs/v2
SDE=/home/francis/tools/sde/sde64
B=$L/bin; mkdir -p $B
cd $W/internal/kernelprint
T0=$(date +%s)
hb() { echo "### $(date -Is) +$(( $(date +%s) - T0 ))s $* | load $(cut -d' ' -f1 /proc/loadavg)"; }
COVPKG=github.com/townsendmerino/aikit/linalg,github.com/townsendmerino/aikit/embed
hb "build: plain and coverage test binaries, no tags and aikit_noavx512"
go test -c -o $B/kp.test . && go test -c -tags aikit_noavx512 -o $B/kp-noavx512.test . &&
go test -c -cover -covermode=set -coverpkg=$COVPKG -o $B/kpcov.test . &&
go test -c -cover -covermode=set -coverpkg=$COVPKG -tags aikit_noavx512 -o $B/kpcov-noavx512.test . || { hb "BUILD FAILED"; exit 1; }
hb "native control: K1 + K3 by the committed path (go test child)"
GOINFER_KERNELPRINT_OUT=$L/fp-native.json timeout 10m $B/kp.test -test.v -test.count=1 -test.run '^(TestCompute_stable|TestCompute_child|TestCompute_reachesCensus)$' > $L/native.log 2>&1
hb "end native rc=$? $(grep -E -- '^--- (PASS|FAIL|SKIP)' $L/native.log | tr '\n' ' ') $(grep -o '[0-9]* census sites.*' $L/native.log)"
k1() { # label chip bin run#
  hb "start K1 $1 run $4 (-$2)"
  GOINFER_KERNELPRINT_OUT=$L/fp-$1-$4.json timeout 20m $SDE -$2 -- $3 -test.v -test.count=1 -test.timeout 18m -test.run '^(TestCompute_stable|TestCompute_child)$' > $L/k1-$1-$4.log 2>&1
  hb "end K1 $1 run $4 rc=$? $(grep -E -- '^--- (PASS|FAIL|SKIP)' $L/k1-$1-$4.log | tr '\n' ' ') $(grep -o '[0-9]* kernels under.*' $L/k1-$1-$4.log)"
}
k3() { # label chip bin covbin run#
  hb "start K3 $1 run $5 (-$2)"
  SDEHARNESS_SDE=$SDE SDEHARNESS_CHIP=$2 SDEHARNESS_COVBIN=$4 timeout 20m $SDE -no-follow-child -$2 -- $3 -test.v -test.count=1 -test.timeout 18m -test.run '^TestCompute_reachesCensus$' > $L/k3-$1-$5.log 2>&1
  hb "end K3 $1 run $5 rc=$? $(grep -E -- '^--- (PASS|FAIL|SKIP)' $L/k3-$1-$5.log | tr '\n' ' ') | $(grep -o 'child config: .*' $L/k3-$1-$5.log) | $(grep -o '[0-9]* census sites.*' $L/k3-$1-$5.log)"
}
k3 hsw-control hsw $B/kp.test $B/kpcov.test 1
for r in 1 2; do
  k1 icx icx $B/kp.test $r;           k3 icx icx $B/kp.test $B/kpcov.test $r
  k1 spr spr $B/kp.test $r;           k3 spr spr $B/kp.test $B/kpcov.test $r
  k1 icx-noavx512 icx $B/kp-noavx512.test $r; k3 icx-noavx512 icx $B/kp-noavx512.test $B/kpcov-noavx512.test $r
  k1 spr-noavx512 spr $B/kp-noavx512.test $r; k3 spr-noavx512 spr $B/kp-noavx512.test $B/kpcov-noavx512.test $r
done
k1 clx clx $B/kp.test 1; k3 clx clx $B/kp.test $B/kpcov.test 1
k1 skx skx $B/kp.test 1; k3 skx skx $B/kp.test $B/kpcov.test 1
hb "all done"
