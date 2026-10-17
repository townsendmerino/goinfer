#!/bin/bash
# The Nemotron-H 4B config fixes (owner, 2026-10-16): build and test them, prove each test can go red, take the parity refresh as a patch, and run the
# 4B CPU load/generate check on the UNMODIFIED checkpoint directory (no config workaround). Runs with nobody watching, in a detached worktree at REV.
#   python3 scripts/night.py add op2b-4b-fixes-cuda --est 30 --by "<who>, OP2b CUDA (4B fixes)" --doc docs/measurements/op2b-cuda-2026-10-16/README.md \
#     -- env REV=<the fixes commit> bash docs/measurements/op2b-cuda-2026-10-16/run-4b-fixes-night.sh
#   DRY=1 ...   stops after the worktree and the checkpoint checks.
# (a) decoder.loadConfig accepts bare Infinity, -Infinity and NaN outside strings; request bodies stay strict.
# (b) the nemotron_h adapter reads attention_head_dim when present.
# Plants, each must turn its test red: (b) the adapter ignores attention_head_dim; (a) loadConfig skips the lenient pass; (a, strictness) the request
# decoder made lenient. A skip counts as a failure. Not a timed run, but it holds the lock (night.py) so nothing shares the box while it builds.
set -uo pipefail
SRC=${SRC:-$HOME/mycode/goinfer}
REV=${REV:?REV is required: the commit with the fixes}
B=${B:-$HOME/goinfer-bench/op2b-cuda-2026-10}
OUT=${OUT:-$HOME/goinfer-logs/op2b-4b-fixes-$(date +%F-%H%M)}
WT=$B/wt-4b-fixes
CK=$HOME/models/nemotron-h-4b-instruct-128k
export PATH=/usr/local/go/bin:$PATH GOWORK=off
mkdir -p "$B" "$OUT" || exit 2
[ -f "$CK/config.json" ] && grep -q '^ *Infinity' "$CK/config.json" || { echo "FATAL: $CK is missing or its config.json no longer carries the bare Infinity this job exists for"; exit 2; }
cd "$SRC" && git fetch -q origin
rev=$(git rev-parse --verify "$REV^{commit}") || { echo "FATAL: $REV is not a commit here"; exit 2; }
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$rev" || exit 2
{ echo "rev: $rev"; echo "started: $(date '+%F %T %Z')"; echo "load: $(uptime)"; } | tee "$OUT/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the worktree and the checkpoint check"; cd "$SRC" && git worktree remove --force "$WT"; exit 0; }
cd "$WT" || exit 2
rc=0
step() { # step <name> <log> <command...>: run; red unless rc 0
  local n=$1 l=$2; shift 2; echo "== $n: $(date '+%T')"
  if "$@" > "$l" 2>&1; then echo "   PASS $n"; else echo "   FAIL $n: $l"; rc=1; fi
}
plant() { # plant <name> <file> <python edit expr: old|||new> <test pkg> <-run regexp>: apply, the test must FAIL, restore
  local n=$1 f=$2 edit=$3 pkg=$4 re=$5
  python3 - "$f" "$edit" <<'PY' || { echo "   PLANT NOT APPLIED $n"; rc=1; return; }
import sys
f, e = sys.argv[1], sys.argv[2]
old, new = e.split("|||")
s = open(f).read()
assert s.count(old) == 1, (f, old, s.count(old))
open(f, "w").write(s.replace(old, new))
PY
  if go test -count=1 -run "$re" "$pkg" > "$OUT/plant-$n.log" 2>&1; then echo "   PLANT $n: the test STAYED GREEN (it cannot see this defect)"; rc=1
  else grep -q -- "--- FAIL" "$OUT/plant-$n.log" && echo "   PLANT $n: red, as it must be" || { echo "   PLANT $n: failed without a test failure (build error?) $OUT/plant-$n.log"; rc=1; }; fi
  git checkout -q -- "$f"
}
step "gofmt" "$OUT/gofmt.log" bash -c 'test -z "$(gofmt -l decoder internal/serveapp)"'
step "vet" "$OUT/vet.log" go vet ./decoder/ ./internal/serveapp/
step "tests (a), (b) and the strictness test" "$OUT/tests.log" go test -count=1 -v -run 'TestLenientConfigJSON|TestLoadConfig_realNemotronH4B|TestNemotronH_attentionHeadDim|TestRequestJSON_nonFiniteStaysRejected' ./decoder/ ./internal/serveapp/
for t in TestLenientConfigJSON_tokens TestLenientConfigJSON_cleanInputUntouched TestLoadConfig_realNemotronH4B TestNemotronH_attentionHeadDim TestRequestJSON_nonFiniteStaysRejected; do
  grep -q -- "--- PASS: $t " "$OUT/tests.log" || { echo "   no '--- PASS: $t' (a skip is not a pass)"; rc=1; }
done
plant b-ignores-attention_head_dim decoder/registry.go $'if cfg.AttentionHeadDim > 0 {|||if false && cfg.AttentionHeadDim > 0 {' ./decoder/ '^TestNemotronH_attentionHeadDim$'
plant a-loadConfig-skips-the-lenient-pass decoder/config.go $'b = lenientConfigJSON(b)|||_ = lenientConfigJSON' ./decoder/ '^TestLoadConfig_realNemotronH4B$'
plant a-request-decoder-lenient internal/serveapp/helpers.go $'if err := json.NewDecoder(r.Body).Decode(v); err != nil {|||var bb bytes.Buffer; _, _ = bb.ReadFrom(r.Body); if err := json.Unmarshal(bytes.ReplaceAll(bytes.ReplaceAll(bb.Bytes(), []byte("Infinity"), []byte("1e308")), []byte("NaN"), []byte("null")), v); err != nil {' ./internal/serveapp/ '^TestRequestJSON_nonFiniteStaysRejected$'
step "the tests again, restored" "$OUT/tests-restored.log" go test -count=1 -run 'TestLenientConfigJSON|TestLoadConfig_realNemotronH4B|TestNemotronH_attentionHeadDim|TestRequestJSON_nonFiniteStaysRejected' ./decoder/ ./internal/serveapp/
step "touched packages: decoder config, features and nemotron tests" "$OUT/decoder-related.log" go test -count=1 -run 'Config|Nemotron|Architecture|Features|Registry|Admission|OptionGrid' ./decoder/
# the parity refresh as a patch: it rewrites testdata/parity_manifest.json in this worktree; the patch is applied by hand (this job never pushes)
step "parity refresh" "$OUT/parity-refresh.log" bash scripts/refresh_parity_hashes.sh
git diff -- testdata/parity_manifest.json > "$OUT/parity-manifest.patch"; echo "   parity manifest patch: $(wc -l < "$OUT/parity-manifest.patch") lines at $OUT/parity-manifest.patch"
git checkout -q -- testdata/parity_manifest.json
# the 4B on the CPU, unmodified directory: serve it, send two greedy prompts
echo "== the 4B on the CPU, unmodified checkpoint: $(date '+%T')"
go build -o "$OUT/serve-cpu" ./cmd/serve || { echo "FATAL: serve build failed"; rc=1; }
if [ -x "$OUT/serve-cpu" ]; then
  port=$((18700 + RANDOM % 200))
  "$OUT/serve-cpu" -model n4="$CK" -backend cpu -quant int4 -addr 127.0.0.1:$port > "$OUT/serve-4b.log" 2>&1 &
  spid=$!
  for _ in $(seq 1 180); do curl -s -o /dev/null http://127.0.0.1:$port/v1/models && break; kill -0 $spid 2>/dev/null || break; sleep 5; done
  if kill -0 $spid 2>/dev/null; then
    for q in "The capital of France is" "Write one sentence about the ocean."; do
      curl -s http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"n4\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":40,\"temperature\":0}" >> "$OUT/4b-replies.jsonl"; echo >> "$OUT/4b-replies.jsonl"
    done
    kill $spid; wait $spid 2>/dev/null
    python3 - "$OUT/4b-replies.jsonl" <<'PY' || rc=1
import json, sys
ok = True
for l in open(sys.argv[1]):
    if not l.strip(): continue
    d = json.loads(l); t = d["choices"][0]["message"]["content"]; n = d["usage"]["completion_tokens"]
    print("   reply (%d tokens): %r" % (n, t[:160])); ok = ok and n >= 8 and len(t.strip()) > 0
sys.exit(0 if ok else 1)
PY
  else echo "   FAIL: the 4B did not load (see $OUT/serve-4b.log)"; tail -3 "$OUT/serve-4b.log"; rc=1; fi
fi
echo "RESULT: $([ $rc = 0 ] && echo 'ALL PASS' || echo 'SOMETHING FAILED') (rc=$rc)" | tee -a "$OUT/provenance.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc
