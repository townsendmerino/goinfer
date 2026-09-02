#!/usr/bin/env python3
"""Remap cmd/gate/gpu.go citations across docs after an edit to that file.

Uses `git diff -U0 <base> <head>` hunk headers rather than the lint's suggested
line numbers, because many lines in that file share a non-discriminating 88-char
anchor (`_, cr, out := g.run(cell{` appears in every cell) and the lint resolves
them all to the first match -- pointing citations at unrelated cells.
"""
import subprocess,re,glob,sys
base=sys.argv[1] if len(sys.argv)>1 else "HEAD~1"
d=subprocess.run(["git","diff","-U0",base,"HEAD","--","cmd/gate/gpu.go"],capture_output=True,text=True).stdout
hunks=[(int(m.group(1)),int(m.group(2) or 1),int(m.group(3)),int(m.group(4) or 1))
       for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',d,re.M)]
def mapline(o):
    delta=0
    for os_,ol,ns,nl in hunks:
        if os_+ol<=o: delta += nl-ol
        elif os_<=o<os_+ol: return o+delta
    return o+delta
tot=0
for p in glob.glob('docs/**/*.md',recursive=True):
    s=open(p,encoding='utf-8').read()
    if 'cmd/gate/gpu.go:' not in s: continue
    def repl(m):
        a=int(m.group(1)); b=m.group(2)
        return f"cmd/gate/gpu.go:{mapline(a)}-{mapline(int(b))}" if b else f"cmd/gate/gpu.go:{mapline(a)}"
    new,n=re.subn(r'cmd/gate/gpu\.go:(\d+)(?:-(\d+))?',repl,s)
    if n: open(p,'w',encoding='utf-8').write(new); print(f"  {p}: {n}"); tot+=n
print(f"remapped {tot}")
