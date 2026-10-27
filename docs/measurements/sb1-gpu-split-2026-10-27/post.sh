#!/bin/bash
# post.sh: the steps after apply.sh that the rewriting tool does not do.
set -e
cd ~/goinfer-bench/nobara-sb01-wt/gpu
for f in residency.go kv_slots.go mamba_layersweep_test.go moe_expert_test.go; do
python3 - $f <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
lines=s.split("\n")
i=next(k for k,l in enumerate(lines) if l.startswith("//go:build"))
lines.insert(i+1,"\n//lint:file-ignore SA1019 Release is Close's deprecated alias, called here as before; it is deprecated for new code only.")
open(p,"w").write("\n".join(lines))
PY
done
PATH=/usr/local/go/bin:$PATH gofmt -w .
# The source guards in n_batch_test.go read the files they guard: two of those now live in internal/kernels, and the
# Reset guard's pattern follows the accessor form residency.go now writes its queue through.
python3 - <<'PY'
p="n_batch_test.go"; s=open(p).read()
for old,new in [('srcOf(t, "attention.go")','srcOf(t, "internal/kernels/attention.go")'),
                ('srcOf(t, "decoderunner.go")','srcOf(t, "internal/kernels/decoderunner.go")'),
                ('"rd.c.queue.WriteBuffer("','"(*kernels.ContextQueue(rd.c)).WriteBuffer("')]:
    assert s.count(old)==1, old
    s=s.replace(old,new)
open(p,"w").write(s)
PY
# The moved tests read the repository's testdata two directories further up.
python3 - <<'PY'
import glob
n=0
for p in glob.glob("internal/kernels/*_test.go"):
    s=open(p).read()
    if '"../testdata' in s:
        n+=s.count('"../testdata')
        s=s.replace('"../testdata','"../../../testdata')
        open(p,"w").write(s)
print("testdata paths moved:",n)
PY
