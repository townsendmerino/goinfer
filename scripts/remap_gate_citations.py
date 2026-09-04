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
import subprocess, re, glob, sys

HUNK_RE = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', re.M)


def parse_hunks(diff_text):
    """Extract (old_start, old_len, new_start, new_len) from a -U0 unified diff's headers."""
    return [
        (int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1))
        for m in HUNK_RE.finditer(diff_text)
    ]


def build_mapper(hunks):
    """Return a function mapping an old-file line number to its new-file line number.

    V-26 (docs/review-2026-09-04.md): a PURE INSERTION hunk (old_len == 0) touches NO old
    lines -- `@@ -N,0 +M,K @@` means "insert K new lines after old line N", and old line N
    itself is untouched. The original condition (`old_start+old_len<=o`) collapses to
    `old_start<=o` when old_len is 0, which wrongly includes o == old_start in "this hunk is
    entirely before o" -- shifting a citation sitting on the line IMMEDIATELY BEFORE a pure
    insertion into the newly inserted text. Old lines strictly AFTER old_start still shift by
    the full insertion size; old_start itself, and anything before it, must not move for
    THIS hunk.
    """
    def mapline(o):
        delta = 0
        for old_start, old_len, new_start, new_len in hunks:
            if old_len == 0:
                if old_start < o:
                    delta += new_len - old_len
            elif old_start + old_len <= o:
                delta += new_len - old_len
            elif old_start <= o < old_start + old_len:
                return o + delta
        return o + delta
    return mapline


def remap_path(git_base, worktree, path):
    """Remap every doc citation of `path` after the edit between git_base and (worktree|HEAD).

    Returns the total citation+anchor count remapped.
    """
    rev = [git_base] if worktree else [git_base, "HEAD"]
    diff = subprocess.run(["git", "diff", "-U0"] + rev + ["--", path],
                           capture_output=True, text=True).stdout
    mapline = build_mapper(parse_hunks(diff))
    pat = re.compile(re.escape(path) + r':(\d+)(?:-(\d+))?')
    # Markdown links carry the SAME line numbers a second time, as a #L anchor
    # (`[decoder/model.go:545-655](../decoder/model.go#L545-L655)`). Remapping only the
    # label leaves the link pointing somewhere else, which is worse than leaving both
    # stale -- the citation lint reads the label and cannot see the disagreement.
    #
    # V-26: this used to be named `base`, the SAME name the caller's git ref parameter used
    # (a plain module-level loop reassigned it once per path) -- so from the second path
    # onward, `rev` above was built from the PREVIOUS path's file basename instead of the git
    # ref, `git diff` got a nonexistent revision, and nothing after the first path was ever
    # remapped. Scoped to this function and given its own name, it can no longer collide.
    file_basename = path.split('/')[-1]
    anchor = re.compile(
        r'((?:\.\.?/)[^)\s]*' + re.escape(path) + r'|(?:\.\.?/)[^)\s]*' + re.escape(file_basename) + r')'
        r'#L(\d+)(?:-L(\d+))?')

    def repl(m):
        a = int(m.group(1))
        b = m.group(2)
        return f"{path}:{mapline(a)}-{mapline(int(b))}" if b else f"{path}:{mapline(a)}"

    def arepl(m):
        pre = m.group(1)
        a = int(m.group(2))
        b = m.group(3)
        return f"{pre}#L{mapline(a)}-L{mapline(int(b))}" if b else f"{pre}#L{mapline(a)}"

    total = 0
    for p in glob.glob('docs/**/*.md', recursive=True):
        s = open(p, encoding='utf-8').read()
        if path + ':' not in s and file_basename + '#L' not in s:
            continue
        new, n = pat.subn(repl, s)
        new, na = anchor.subn(arepl, new)
        if n or na:
            open(p, 'w', encoding='utf-8').write(new)
            print(f"  {p}: {n} citation(s) + {na} anchor(s) ({path})")
            total += n + na
    return total


def main(argv):
    args = argv[1:]
    worktree = bool(args) and args[0] == "--worktree"
    if worktree:
        args = args[1:]
    git_base = args[0] if args else "HEAD~1"
    paths = args[1:] or ["cmd/gate/gpu.go"]
    tot = sum(remap_path(git_base, worktree, path) for path in paths)
    print(f"remapped {tot}")


if __name__ == "__main__":
    main(sys.argv)
