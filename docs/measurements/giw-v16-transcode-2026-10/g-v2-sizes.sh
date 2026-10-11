#!/bin/bash
# G-V2: the size of a v15 and a v16 transcode of one source, per target. Writes one file at a time and removes it.
B=~/goinfer-bench/v16-sizes
for m in "$@"; do
  src=~/models/$m
  for cfg in "metal -embed-int4" "metal" "cpu-arm64 -embed-int4" "cpu-arm64" "cuda"; do
    tgt=${cfg%% *}; extra=""; [ "$cfg" != "$tgt" ] && extra=${cfg#* }
    line="$m target=$tgt ${extra:-plain}"
    for v in v15 v16; do
      out=$B/out.$v.giw
      if ! $B/prequant-$v -quant int4 $extra -target $tgt -o $out $src > $B/last.$v.log 2>&1; then line="$line $v=FAILED"; continue; fi
      line="$line $v=$(stat -f %z $out)"
      find $out $out.verified -delete 2>/dev/null
    done
    echo "$line"
  done
done
