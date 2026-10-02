#!/bin/bash
# D10 reference fixture (docs/measurements/decisions-d10-clef-2026-10-02.md): Cloudflare's own code on Cloudflare/clef-flash (pinned release in the archive),
# the 150 D0 items as Clef records. Unattended: a PINNED copy of scripts/pin_clef_d10.py and of records.jsonl (sha256 checked), output to a durable directory,
# resumable (rows already written are skipped), no prompts. Fixture generation, not a timed measurement.
#   DTYPE=f32 HIDDEN=3 bash run-clef-fixture.sh        all 150 items at f32 (~38 GB RAM) plus last_hidden_state for the 3 shortest items; est 50 min
#   DTYPE=bf16 PART=1 bash run-clef-fixture.sh         bf16 (the release's dtype), the first 75 rows; est 95 min  (bf16 is ~5x slower than f32 on this CPU: no bf16 hardware)
#   DTYPE=bf16 PART=2 bash run-clef-fixture.sh         bf16, the next 75 rows
set -euo pipefail
B=${B:-$HOME/goinfer-bench/decisions-d10}
PY=${PY:-$HOME/d0venv/bin/python3}
SCRIPT=$B/pin_clef_d10.py;  SCRIPT_SHA=caebd39a836e31463463f60eb6225a80e7db0f265de418a88a994afae5222b93
RECORDS=$B/out/records.jsonl; RECORDS_SHA=5942ccb997012d69c37945ef8cf3a2dd1974f62009ef14d9a668f5d4c49f9b7c
DTYPE=${DTYPE:?f32 or bf16}
chk() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "$1: sha256 differs from the registered one"; exit 1; }; }
chk "$SCRIPT" "$SCRIPT_SHA"; chk "$RECORDS" "$RECORDS_SHA"
for f in model-00001-of-00004 model-00002-of-00004 model-00003-of-00004 model-00004-of-00004; do [ -s /srv/models/clef-flash/$f.safetensors ] || { echo "missing shard $f"; exit 1; }; done
args=(--dtype "$DTYPE")
[ -z "${HIDDEN:-}" ] || args+=(--hidden "$HIDDEN")
case "${PART:-}" in 1|2) args+=(--limit 75);; "") ;; *) echo "PART is 1 or 2"; exit 1;; esac
OUT=$B/out; n0=$(cat "$OUT/probs_$DTYPE.jsonl" 2>/dev/null | wc -l)
[ "${PART:-}" != "2" ] || [ "$n0" -ge 75 ] || { echo "PART=2 needs the 75 rows of PART=1 first (have $n0)"; exit 1; }
echo "$(date +%T) == D10 fixture $DTYPE start (part ${PART:-all}); rows already written: $n0; load $(cut -d' ' -f1-3 /proc/loadavg); free $(free -g | awk '/Mem/{print $7}') GB"
D10_OUT="$OUT" CLEF_MODEL=/srv/models/clef-flash "$PY" -u "$SCRIPT" model "${args[@]}"
n=$(wc -l < "$OUT/probs_$DTYPE.jsonl")
want=150; [ "${PART:-}" = "1" ] && want=75
echo "$(date +%T) == done; rows $n (wanted $want)"
[ "$n" -ge "$want" ] || { echo "only $n of $want rows"; exit 1; }
echo "$(date +%T) == DONE; results in $OUT"
