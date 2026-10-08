#!/usr/bin/env bash
# S9 on CUDA part B, G3q (docs/tasks/task-multimodal-support-2026-10.md, "S9 on CUDA, part B"): the real Gemma 4 E2B image turn served on CUDA, prefill resident
# (the NEW binary) against today's CPU prefill + upload (the pre-change binary), greedy 32 tokens with top-3 logprobs, replies graded by the registered near-tie rule.
# Four runs of run-gs3c-served.sh (one fresh server per arm), all with GS3C_EXTRA="--kv-sessions 1 -ctx 4096 --vision ~/models/gemma-4-E2B-unq" as S4's served reads used:
#   old        pre-change binary, tower on the CPU      -> the reference (CPU tower + CPU prefill + upload)
#   new        new binary, tower on the CPU             -> resident prefill, same features as the reference
#   new-ctl    new binary, GOINFER_BATCHED_PREFILL=0    -> the new path declines into the unchanged bridge: a same-binary control
#   new-auto   new binary, tower on CUDA                -> resident prefill with the CUDA tower (features differ ~1e-6 from the CPU tower: near-ties expected)
# Usage (from the repo root): run-s9b-served.sh <new serve binary> <old serve binary> <out dir> [model]
set -uo pipefail
NEW=${1:?}; OLD=${2:?}; OUT=${3:?}; MODEL=${4:-$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf}
[ -f testdata/glm_ocr/table.png ] || { echo "run from the repo root" >&2; exit 2; }
mkdir -p "$OUT"
S=docs/measurements/multimodal-support-2026-10/run-gs3c-served.sh
VISION=${VISION:-$HOME/models/gemma-4-E2B-unq}
[ -d "$VISION" ] || { echo "FATAL: no vision checkpoint at $VISION" >&2; exit 2; }
export GS3C_EXTRA="--kv-sessions 1 -ctx 4096 --vision $VISION"
{ echo "new: $NEW ($(cat "$(dirname "$NEW")/rev" 2>/dev/null))"; echo "old: $OLD ($(cat "$(dirname "$OLD")/rev.old" 2>/dev/null))"; echo "model: $MODEL"
  echo "started: $(date '+%F %T %Z')"; echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"; } | tee "$OUT/provenance.txt"
rc=0
bash $S "$OLD" "$OUT/old"      cuda:cpu  "$MODEL" > "$OUT/old.console" 2>&1      || { echo "!! old run failed"; rc=1; }
bash $S "$NEW" "$OUT/new"      cuda:cpu  "$MODEL" > "$OUT/new.console" 2>&1      || { echo "!! new run failed"; rc=1; }
GOINFER_BATCHED_PREFILL=0 bash $S "$NEW" "$OUT/new-ctl" cuda:cpu "$MODEL" > "$OUT/new-ctl.console" 2>&1 || { echo "!! new-ctl run failed"; rc=1; }
bash $S "$NEW" "$OUT/new-auto" cuda:auto "$MODEL" > "$OUT/new-auto.console" 2>&1 || { echo "!! new-auto run failed"; rc=1; }
python3 - "$OUT" <<'PY' || rc=1
import glob, json, math, re, sys
out = sys.argv[1]
def one(run, pat):
    f = glob.glob(f"{out}/{run}/gs3c-logprobs-*.json")
    return json.load(open(f[0])) if f else None
def log(run):
    f = [x for x in glob.glob(f"{out}/{run}/gs3c-*.log")]
    return open(f[0]).read() if f else ""
ref = one("old", None); ok = ref is not None
if ref is None: print("!! no reference logprobs")
for run, want_resident in (("new", True), ("new-ctl", False), ("new-auto", True)):
    lp, lg = one(run, None), log(run)
    resident = "(prefill resident)" in lg
    dec = "decoded" in lg and "resident path" in lg
    print(f"== {run}: image prefill resident line present: {resident} (want {want_resident}); decode on the resident path: {dec}")
    if resident != want_resident or not dec: ok = False
    if lp is None or ref is None: ok = False; continue
    a, b = lp, ref
    k = next((i for i in range(min(len(a), len(b))) if a[i]["token"] != b[i]["token"]), None)
    if k is None and len(a) == len(b):
        same = a == b
        print(f"   {len(a)} tokens, reply IDENTICAL to the reference" + ("" if same else "; logprob values differ"))
        continue
    if k is None: print("   reply length differs"); ok = False; continue
    ptop = math.exp(b[k]["top_logprobs"][0]["logprob"])
    other = next((math.exp(t["logprob"]) for t in b[k]["top_logprobs"] if t["token"] == a[k]["token"]), 0.0)
    tie = other >= ptop / 2
    print(f"   first differing generated token {k}: reference {b[k]['token']!r} (p={ptop:.3f}), this arm {a[k]['token']!r} (p={other:.3f}); near-tie (p(other) >= half p(top)): {tie}")
    ok = ok and tie
print("G3q:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
for r in old new new-ctl new-auto; do echo "-- $r:"; grep -hE "^  cuda[a-z0-9-]* \(" "$OUT/$r.console" | cut -c1-170; done
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc
