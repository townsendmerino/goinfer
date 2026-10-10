#!/usr/bin/env python3
"""Tooling for the Go comment diet (docs/tasks/task-code-comments-2026-10.md, CC1/CC4).

  comment_diet.py blocks FILE [MINLEN]     list the comment blocks of FILE (default MINLEN 6): line range, length, whether the block
                                           carries a history marker, and the declaration under it. Comments inside a raw string
                                           (an embedded MSL/CUDA/WGSL kernel) are not comments and are not listed.
  comment_diet.py census FILE...           comment lines and history-marked comment lines per file
  comment_diet.py pointers                 check every `docs/code-notes/<pkg>.md#Heading` (or `<pkg>.md, "Heading"`) pointer in the Go
                                           comments of the tree against the headings of that notes file; exit 1 on an unresolved one
  comment_diet.py apply PLAN.py            replace comment blocks and move the removed text verbatim to docs/code-notes/<pkg>.md

A plan is a Python file defining FILE (repo-relative .go path), optionally NOTES (default docs/code-notes/<dir with / as ->.md) and
DATE, and EDITS: a list of dicts
    dict(start=L1, end=L2, new=['line', ...], heading='Decl' | None, where='above `Decl`')
`start`/`end` are the block's line range in the file AS IT STANDS when apply runs; edits are applied bottom-up, so a plan's numbers are all
those of the current file. `new` is the replacement text without the leading `// ` (an empty string is a bare `//`); the block's indent is
kept. With a `heading` the OLD block is appended, unedited, to the notes file under `## <heading>` after a line naming the file, where it
came from and the date; with heading=None nothing moves (use it when the new text keeps everything the old said). A heading used twice
is refused: give each moved block its own. Moved text must not contain `file.go:NNN`: the citation lint is red on it, so name the declaration
in the notes file by hand after applying (the one allowed edit to moved text).

Always follow an apply with `go run ./cmd/gate comments-only HEAD --worktree`."""
import os
import re
import runpy
import sys

MARK = re.compile(r"\b20\d\d-\d\d|\b[0-9a-f]{7,10}\b(?=[ ,)])|\b[MRPCDGTB]-?\d{1,3}[a-z]?\b|\bdocs/|\btask-|"
                  r"\d+(\.\d+)?\s?(ms|µs|us|x|×|%|tok/s|GB|MB)\b|used to|landed|shipped|found by|measured|STILL OPEN|whoever|HISTORY|originally|previously")
DIRECTIVE = ("//go:", "// +build", "//+build", "//nolint", "//lint:ignore")


def comment_lines(src: str):
    """{1-based line -> True} for every line that is wholly a `//` comment line outside a raw string / block comment."""
    flags = {}
    i, n, line = 0, len(src), 1
    state = None  # None | 'raw' | 'block'
    line_start = True
    only_ws = True  # nothing but whitespace seen on this line so far
    while i < n:
        c = src[i]
        if state == 'raw':
            if c == '`':
                state = None
            if c == '\n':
                line += 1
                only_ws = True
            i += 1
            continue
        if state == 'block':
            if src.startswith('*/', i):
                state = None
                i += 2
                continue
            if c == '\n':
                line += 1
                only_ws = True
            i += 1
            continue
        if c == '\n':
            line += 1
            only_ws = True
            i += 1
            continue
        if src.startswith('//', i):
            if only_ws:
                flags[line] = True
            j = src.find('\n', i)
            i = n if j < 0 else j
            continue
        if src.startswith('/*', i):
            state = 'block'
            i += 2
            only_ws = False
            continue
        if c == '`':
            state = 'raw'
        elif c == '"' or c == "'":
            q = c
            i += 1
            while i < n and src[i] != q and src[i] != '\n':
                i += 2 if src[i] == '\\' else 1
        if not c.isspace():
            only_ws = False
        i += 1
    return flags


def line_comments(src: str):
    """{1-based line -> comment text} for every `//` comment on a line, whole-line or trailing, outside raw strings, strings and block
    comments. A `/* */` comment is not reported (the tree uses `//`)."""
    out = {}
    i, n, line = 0, len(src), 1
    state = None
    while i < n:
        c = src[i]
        if state == 'raw':
            if c == '`':
                state = None
            elif c == '\n':
                line += 1
            i += 1
            continue
        if state == 'block':
            if src.startswith('*/', i):
                state = None
                i += 2
                continue
            if c == '\n':
                line += 1
            i += 1
            continue
        if c == '\n':
            line += 1
        elif src.startswith('//', i):
            j = src.find('\n', i)
            j = n if j < 0 else j
            out[line] = src[i:j]
            i = j
            continue
        elif src.startswith('/*', i):
            state = 'block'
            i += 2
            continue
        elif c == '`':
            state = 'raw'
        elif c == '"' or c == "'":
            q = c
            i += 1
            while i < n and src[i] != q and src[i] != '\n':
                i += 2 if src[i] == '\\' else 1
        i += 1
    return out


def blocks(src: str):
    lines = src.split('\n')
    flags = comment_lines(src)
    out, i = [], 0
    while i < len(lines):
        ln = i + 1
        if flags.get(ln) and not lines[i].lstrip().startswith(DIRECTIVE):
            j = i
            while flags.get(j + 2) and not lines[j + 1].lstrip().startswith(DIRECTIVE):
                j += 1
            out.append((i + 1, j + 1))
            i = j + 1
        else:
            i += 1
    return lines, out


def cmd_blocks(path, minlen=6):
    src = open(path).read()
    lines, bl = blocks(src)
    tot = sum(b - a + 1 for a, b in bl)
    print(f"{path}: {len(lines)} lines, {tot} comment lines in {len(bl)} block(s)")
    for a, b in bl:
        if b - a + 1 < minlen:
            continue
        txt = ' '.join(lines[a - 1:b])
        nxt = next((lines[k].strip()[:70] for k in range(b, min(b + 3, len(lines))) if lines[k].strip() and not lines[k].lstrip().startswith('//')), '')
        print(f"{a}-{b} ({b - a + 1}){' H' if MARK.search(txt) else '  '} -> {nxt}")


def cmd_census(paths):
    tc = tm = 0
    for p in paths:
        src = open(p).read()
        lines, bl = blocks(src)
        c = sum(b - a + 1 for a, b in bl)
        m = sum(1 for a, b in bl for k in range(a - 1, b) if MARK.search(lines[k]))
        tc, tm = tc + c, tm + m
        print(f"{p}: {c} comment lines, {m} with a history marker")
    print(f"total: {tc} comment lines, {tm} with a history marker")


def cmd_apply(plan_path):
    plan = runpy.run_path(plan_path)
    file = plan['FILE']
    notes = plan.get('NOTES') or 'docs/code-notes/' + os.path.dirname(file).replace('/', '-') + '.md'
    date = plan.get('DATE', '2026-10-09')
    pkg = os.path.dirname(file)
    src = open(file).read()
    _, valid = blocks(src)
    validset = {(a, b) for a, b in valid}
    lines = src.split('\n')
    moved, headings = [], set()
    for e in sorted(plan['EDITS'], key=lambda e: -e['start']):
        a, b = e['start'], e['end']
        # the edit must lie inside one real comment block (not a raw string, not a directive)
        if not any(s <= a and b <= t for s, t in validset):
            sys.exit(f"{file}:{a}-{b} is not inside a comment block (or touches a directive / raw string)")
        old = lines[a - 1:b]
        indent = re.match(r'\s*', old[0]).group(0)
        new = [(indent + '// ' + t if t else indent + '//') for t in e['new']]
        if e.get('heading'):
            if e['heading'] in headings:
                sys.exit(f"heading {e['heading']!r} used twice; give each moved block its own")
            headings.add(e['heading'])
            text = []
            for l in old:
                s = l.lstrip()[2:]
                text.append(s[1:] if s.startswith(' ') else s)
            moved.append((a, e['heading'], e.get('where', 'above `' + e['heading'] + '`'), text))
        lines[a - 1:b] = new
    open(file, 'w').write('\n'.join(lines))
    moved.sort()
    if moved:
        existing = open(notes).read() if os.path.exists(notes) else (
            f"# {pkg}: notes moved out of code comments\n\nHistory, measurements and open work that used to sit in the comments of `{pkg}`, moved here verbatim by the comment diet\n"
            "(`docs/tasks/task-code-comments-2026-10.md`, CC1). The code comment above each declaration keeps what the code does, its contract and\n"
            "its guardrails, and points here by heading. This file is off the read path: open it when a pointer sends you. Text is unedited,\n"
            "except that a `file.go:NNN` reference inside it names the declaration instead (the line numbers had already gone stale).\n")
        seen = set(re.findall(r'^## (.+)$', existing, re.M))
        for _, h, where, text in moved:
            if h in seen:
                sys.exit(f"notes already has a section {h!r}: choose another heading")
            seen.add(h)
            existing += f"\n## {h}\n\nMoved from `{file}` (the comment {where}) on {date}.\n\n```text\n" + '\n'.join(text).rstrip() + "\n```\n"
        os.makedirs(os.path.dirname(notes), exist_ok=True)
        open(notes, 'w').write(existing)
        bad = [l for l in existing.split('\n') if re.search(r'\b[\w/]+\.(go|py|sh):\d+', l)]
        if bad:
            print(f"WARNING: {notes} holds {len(bad)} path:line reference(s); name the declaration instead (the citation lint is red on them):")
            for l in bad[:10]:
                print("   ", l.strip()[:140])
    print(f"{file}: {len(plan['EDITS'])} edit(s), {len(moved)} moved block(s) -> {notes}")


def cmd_pointers():
    import glob
    import subprocess
    root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip()
    notes = {}
    for f in glob.glob(os.path.join(root, 'docs/code-notes/*.md')):
        notes[os.path.basename(f)[:-3]] = set(re.findall(r'^## (.+?)\s*$', open(f).read(), re.M))
    ref = re.compile(r'docs/code-notes/([\w-]+)\.md(?:#(\S.*)|,\s*"([^"]+)")?')
    bad, n = [], 0
    files = subprocess.run(['git', 'ls-files', '*.go'], capture_output=True, text=True, cwd=root).stdout.split()
    for rel in files:
        if rel == 'cmd/gate/comments_only_test.go':  # its fixtures name notes files that do not exist
            continue
        for i, line in enumerate(open(os.path.join(root, rel)).read().split('\n'), 1):
            if '//' not in line:
                continue
            for m in ref.finditer(line.split('//', 1)[1]):
                pkg, frag, quoted = m.group(1), m.group(2), m.group(3)
                if frag is None and quoted is None:
                    continue
                n += 1
                heads = notes.get(pkg)
                if heads is None:
                    bad.append(f"{rel}:{i}: no notes file docs/code-notes/{pkg}.md")
                    continue
                want = quoted if quoted is not None else frag.rstrip('.,;:)\'"`')
                if quoted is None:
                    ok = any(want == h or want.startswith(h) and not (want[len(h):len(h) + 1].isalnum() or want[len(h):len(h) + 1] in '_.') for h in heads)
                else:
                    ok = want in heads
                if not ok:
                    bad.append(f"{rel}:{i}: docs/code-notes/{pkg}.md has no heading {want[:60]!r}")
    print(f"{n} pointer(s) checked, {len(bad)} unresolved")
    for b in bad:
        print("  " + b)
    return 1 if bad else 0


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == 'blocks':
        cmd_blocks(a[1], int(a[2]) if len(a) > 2 else 6)
    elif len(a) >= 2 and a[0] == 'census':
        cmd_census(a[1:])
    elif a == ['pointers']:
        sys.exit(cmd_pointers())
    elif len(a) == 2 and a[0] == 'apply':
        cmd_apply(a[1])
    else:
        sys.exit(__doc__)
