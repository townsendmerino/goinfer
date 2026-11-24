#!/usr/bin/env bash
# The row-pad A/B (README beside this file): P1, P2, P3 and the two controls, in that order, each its own A B B A x 5 under the exclusive
# lock. Resumable: each comparison appends to its own cells.tsv and skips the cells it holds. Stops itself at 21:25 PT.
# usage: run-ab.sh <bin dir> <goinfer commit> <out dir>
set -uo pipefail
B=$1 H=$2 O=$3
cd "$(dirname "$0")/../../.."
AB=docs/measurements/cpu-qk-multichain-2026-11-23/run_ab.py
K16=$B/dec-rowpad-K16-$H.test K4=$B/dec-rowpad-K4-$H.test
lock() { python3 scripts/timing_lock.py run --wait 14400 --label "$1" -- "${@:2}"; }
lock rowpad-P1 python3 -u $AB --bin-a $K16 --bin-b $K16 --pad-b --models P3M --depths 3900,128 --out $O/P1 || exit 1
lock rowpad-P2 python3 -u $AB --bin-a $K4 --bin-b $K16 --pad-b --models P3M --depths 3900,128 --out $O/P2 || exit 1
lock rowpad-P3 python3 -u $AB --bin-a $K4 --bin-b $K4 --pad-b --models P3M --depths 3900,128 --out $O/P3 || exit 1
lock rowpad-C1 python3 -u $AB --bin-a $K16 --bin-b $K16 --pad-b --models Q05 --depths 3900 --out $O/C1 || exit 1
lock rowpad-C2 python3 -u $AB --bin-a $K16 --bin-b $K16 --pad-b --kv-a f16 --kv-b f16 --models P3M --depths 3900 --out $O/C2 || exit 1
echo ROWPAD-AB-DONE
