#!/bin/bash
# apply.sh <out dir>: move the cut's files into gpu/internal/kernels (git mv, then the rewritten text) and write the rest.
set -e
O=$1
cd ~/goinfer-bench/nobara-sb01-wt/gpu
mkdir -p internal/kernels
for f in $(ls $O/kernels); do
  if [ -f $f ]; then git mv $f internal/kernels/$f; fi
  cp $O/kernels/$f internal/kernels/$f
done
for f in $(ls $O/gpu); do cp $O/gpu/$f $f; done
git mv writable_other_test.go internal/kernels/writable_other_test.go
sed -i 's/^package gpu$/package kernels/' internal/kernels/writable_other_test.go
