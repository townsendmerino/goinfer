#!/usr/bin/env python3
"""Remap docs citations into a source file after an edit to that file.

Uses `git diff -U0 <base> HEAD -- <path>` hunk headers rather than the lint's suggested
line numbers, because many lines in these files share a non-discriminating 88-char
anchor (`_, cr, out := g.run(cell{` appears in every cell in cmd/gate/gpu.go) and the
lint resolves them all to the first match -- pointing citations at unrelated cells.

    scripts/remap_gate_citations.py <base> [path ...]              # committed edits, <base>..HEAD
    scripts/remap_gate_citations.py --worktree <base> [path ...]   # edits still in the working tree

A citation whose CONTENT the edit deleted is not remappable and is not remapped: the
sentence citing it no longer has a source, which is a prose fix, not a line-number one.
The lint reports those separately as CONTENT ABSENT.
"""
import subprocess,re,glob,sys
args=sys.argv[1:]
worktree = args and args[0]=="--worktree"
if worktree: args=args[1:]
base=args[0] if args else "HEAD~1"
paths=args[1:] or ["cmd/gate/gpu.go"]
tot=0
for path in paths:
    rev=[base] if worktree else [base,"HEAD"]
    d=subprocess.run(["git","diff","-U0"]+rev+["--",path],capture_output=True,text=True).stdout
    hunks=[(int(m.group(1)),int(m.group(2) or 1),int(m.group(3)),int(m.group(4) or 1))
           for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',d,re.M)]
    def mapline(o):
        delta=0
        for os_,ol,ns,nl in hunks:
            if os_+ol<=o: delta += nl-ol
            elif os_<=o<os_+ol: return o+delta
        return o+delta
    pat=re.compile(re.escape(path)+r':(\d+)(?:-(\d+))?')
    for p in glob.glob('docs/**/*.md',recursive=True):
        s=open(p,encoding='utf-8').read()
        if path+':' not in s: continue
        def repl(m):
            a=int(m.group(1)); b=m.group(2)
            return f"{path}:{mapline(a)}-{mapline(int(b))}" if b else f"{path}:{mapline(a)}"
        new,n=re.subn(pat,repl,s)
        if n: open(p,'w',encoding='utf-8').write(new); print(f"  {p}: {n} ({path})"); tot+=n
print(f"remapped {tot}")
