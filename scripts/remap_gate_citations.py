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

RUN IT ONCE PER <base>. It rewrites every doc that cites the path, and it has no way to
tell an already-remapped citation from a stale one -- a second run applies the same delta
again and lands the citation somewhere neither right nor recognisably wrong. Measured
2026-09-02: a second pass over decoder/model.go moved `decoder/model.go:1037` to 1022 and
then to 1007, and the lint's own suggestions were what caught it. If you need to fix up
one document after a run, use the lint's content-anchored suggestions ("the cited content
is now at line N"), not another pass of this.
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
    # Markdown links carry the SAME line numbers a second time, as a #L anchor
    # (`[decoder/model.go:545-655](../decoder/model.go#L545-L655)`). Remapping only the
    # label leaves the link pointing somewhere else, which is worse than leaving both
    # stale -- the citation lint reads the label and cannot see the disagreement.
    base=path.split('/')[-1]
    anchor=re.compile(r'((?:\.\.?/)[^)\s]*'+re.escape(path)+r'|(?:\.\.?/)[^)\s]*'+re.escape(base)+r')#L(\d+)(?:-L(\d+))?')
    for p in glob.glob('docs/**/*.md',recursive=True):
        s=open(p,encoding='utf-8').read()
        if path+':' not in s and base+'#L' not in s: continue
        def repl(m):
            a=int(m.group(1)); b=m.group(2)
            return f"{path}:{mapline(a)}-{mapline(int(b))}" if b else f"{path}:{mapline(a)}"
        def arepl(m):
            pre=m.group(1); a=int(m.group(2)); b=m.group(3)
            return f"{pre}#L{mapline(a)}-L{mapline(int(b))}" if b else f"{pre}#L{mapline(a)}"
        new,n=re.subn(pat,repl,s)
        new,na=anchor.subn(arepl,new)
        if n or na: open(p,'w',encoding='utf-8').write(new); print(f"  {p}: {n} citation(s) + {na} anchor(s) ({path})"); tot+=n+na
print(f"remapped {tot}")
