#!/usr/bin/env bash
# #1160's int4-head check, end to end (docs/measurements/embed-int4-longctx-2026-11-20/README.md). Every phase holds the timing lock:
# CPU work shared, GPU arms exclusive, each waiting for any holder. Stops at the first failed phase.
set -uo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
R=$HOME/mycode/reanchor-wt; W=$HOME/goinfer-work/fid1160; L="python3 $R/scripts/timing_lock.py run --wait 7200"
D=$R/docs/measurements/embed-int4-longctx-2026-11-20; REV=9d7485a29
DOCS="internal/app/serveapp/openai.go backend/cuda/resident.go decoder/weights.go docs/server.md docs/env-vars.md RELEASING.md"
ts() { date '+%F %T %Z'; }
phase() { echo "=== $(ts) START $1 (load $(cut -d' ' -f1 /proc/loadavg))"; P0=$(date +%s); }
done_() { local r=$1; echo "=== $(ts) END $PH rc=$r ($(( $(date +%s) - P0 ))s)"; [ $r = 0 ] || { echo "=== STOP: $PH failed"; exit $r; }; }
PH=build; phase $PH
$L --shared --label 1160-build -- sh -c "cd $R/backend/cuda && go vet -tags 'cuda goinfer_testhooks' . && go test -c -tags 'cuda goinfer_testhooks' -o $W/ei4lc.test ." > $W/logs/build.log 2>&1; done_ $?
for m in 0.5B 1.5B; do
  case $m in 0.5B) G=$HOME/models/qwen05; F=qwen2.5-coder-0.5b-instruct-q4_k_m.gguf;; 1.5B) G=$HOME/models/qwen15; F=qwen2.5-coder-1.5b-instruct-q4_k_m.gguf;; esac
  PH=ref-$m; phase $PH
  if [ -f $W/ref-$m/.done ]; then echo "    reused"; else
    $L --shared --label 1160-hfref -- ~/.venv-vl/bin/python -I $D/ref.py --gguf-dir $G --gguf-file $F --repo $R --rev $REV --out $W/ref-$m $DOCS > $W/logs/ref-$m.log 2>&1 && touch $W/ref-$m/.done
  fi; r=$?; tail -7 $W/logs/ref-$m.log; done_ $r
done
for m in 0.5B 1.5B; do
  case $m in 0.5B) G=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf;; 1.5B) G=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf;; esac
  for arm in cuda-int4head cuda-int8head; do
    PH=$m-$arm; phase $PH
    $L --label 1160-gpu -- env EI4LC_ARM=$arm EI4LC_MODEL=$G EI4LC_REF=$W/ref-$m EI4LC_OUT=$W/results/$m-$arm.jsonl \
      $W/ei4lc.test -test.run '^TestEmbedInt4LongContext_run$' -test.v -test.timeout 30m > $W/logs/$m-$arm.log 2>&1; r=$?
    grep -E "^ei4lc .*(done|resident|cpu)|--- (PASS|FAIL|SKIP)|VOID" $W/logs/$m-$arm.log | tail -9; done_ $r
  done
done
for m in 0.5B 1.5B; do
  case $m in 0.5B) G=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf; DL="";; 1.5B) G=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf; DL="backend_cuda_resident.go,docs_server.md";; esac
  PH=$m-cpu-int4head; phase $PH
  $L --shared --label 1160-cpu -- env EI4LC_ARM=cpu-int4head EI4LC_MODEL=$G EI4LC_REF=$W/ref-$m EI4LC_DOCS="$DL" EI4LC_OUT=$W/results/$m-cpu-int4head.jsonl \
    $W/ei4lc.test -test.run '^TestEmbedInt4LongContext_run$' -test.v -test.timeout 60m > $W/logs/$m-cpu.log 2>&1; r=$?
  grep -E "^ei4lc .*done|--- (PASS|FAIL|SKIP)" $W/logs/$m-cpu.log | tail -7; done_ $r
done
PH=grade; phase $PH
EI4LC_GRADE_DIR=$W/results EI4LC_GRADE_MODELS=0.5B,1.5B $W/ei4lc.test -test.run '^TestEmbedInt4LongContext_grade$' -test.v > $W/logs/grade.log 2>&1; r=$?
grep -E "VERDICT|K=|--- " $W/logs/grade.log; done_ $r
echo "=== $(ts) ALL DONE"
