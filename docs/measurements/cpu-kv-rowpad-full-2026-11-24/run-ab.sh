#!/usr/bin/env bash
# The full row-pad A/B (README beside this file), comparisons in registered order, each its own A B B A x 5 under the exclusive lock.
# Resumable (each comparison appends to its own cells.tsv and skips cells it holds); run_ab.py stops itself at 21:25 PT.
# usage: run-ab.sh <bin dir> <goinfer commit> <out dir>
set -uo pipefail
B=$1 H=$2 O=$3
cd "$(dirname "$0")/../../.."
AB=docs/measurements/cpu-qk-multichain-2026-11-23/run_ab.py
K16=$B/dec-rowpad-K16-$H.test K4=$B/dec-rowpad-K4-$H.test
lock() { python3 scripts/timing_lock.py run --wait 14400 --label "$1" -- "${@:2}"; }
lock rowpad-full-R1 python3 -u $AB --bin-a $K16 --bin-b $K16 --pad-b --models P3M --depths 3900,128 --out $O/R1 || exit 1
lock rowpad-full-R2 python3 -u $AB --bin-a $K4 --bin-b $K16 --pad-a --pad-b --models P3M --depths 3900,128 --out $O/R2 || exit 1
lock rowpad-full-F1 python3 -u $AB --bin-a $K16 --bin-b $K16 --pad-a --kv-b f16 --models P3M --depths 3900 --out $O/F1 || exit 1
lock rowpad-full-F0p python3 -u $AB --bin-a $K16 --bin-b $K16 --widen-a --kv-a f16 --kv-b f16 --models P3M --depths 3900 --out $O/F0 || exit 1
lock rowpad-full-F0q python3 -u $AB --bin-a $K16 --bin-b $K16 --widen-a --kv-a f16 --kv-b f16 --models Q05 --depths 3900,128 --out $O/F0 || exit 1
lock rowpad-full-C1 python3 -u $AB --bin-a $K16 --bin-b $K16 --pad-b --models Q05 --depths 3900 --out $O/C1 || exit 1
echo ROWPAD-FULL-AB-DONE
