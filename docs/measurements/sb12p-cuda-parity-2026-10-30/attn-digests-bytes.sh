#!/usr/bin/env bash
# Throwaway: re-record cuda/kernels' attention digests on this driver (-update-attn-digests) in a scratch worktree, restore the
# committed file, and say whether the digests or only the anchor moved. Usage: attn-digests-bytes.sh OUTPREFIX (WT= the worktree)
set -u
WT=${WT:-$HOME/goinfer-bench/nobara-ci01-ab-041884dbe}; OUT=${1:?out}
cd "$WT/cuda" && cp testdata/attn_digests.json "$OUT.before.json" || exit 2
export PATH=/usr/local/go/bin:$PATH CGO_ENABLED=0
go test -tags cuda -count=1 -run '^TestAttnKernelDigests$' -v ./kernels/ -args -update-attn-digests > "$OUT.log" 2>&1; echo "rc=$?"
cp testdata/attn_digests.json "$OUT.after.json"; git checkout -q -- testdata/attn_digests.json
python3 - "$OUT.before.json" "$OUT.after.json" <<'P'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
print("anchor before:", a["anchor"]); print("anchor after: ", b["anchor"])
da, db = a["digests"], b["digests"]
print("cases", len(da), len(db), "equal" if da == db else "DIFFER")
if da != db and isinstance(da, dict):
    for k in sorted(set(da)|set(db)):
        if da.get(k) != db.get(k): print("  differs:", k)
P
