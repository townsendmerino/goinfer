#!/usr/bin/env python3
"""task_docs.py — rank, check and archive the task docs in docs/tasks/, so closing one out is a command.

    python3 scripts/task_docs.py rank [--top N]        most recently created or worked first
    python3 scripts/task_docs.py check [--strict]      the hygiene a close-out depends on
    python3 scripts/task_docs.py evidence DOC          each item ID the doc names, and the commits that name it
    python3 scripts/task_docs.py archive DOC --status FILE [--refs auto|docs|all|stub] [--apply] [--stage]

WHY THIS EXISTS. Closing a task doc is half judgment (is every item shipped, killed or owned elsewhere?) and half
ritual (`.claude/skills/doc-review/SKILL.md` §3: the ARCHIVED header copied verbatim, a dated status block, `git mv`,
every live reference re-pointed or a pointer stub left, `docs/README.md` recounted, the citation lint run). The ritual
is where the slips are. On 2026-10-21, 11 of the 136 archived docs had no ARCHIVED header, among them the five most
recent archives (2026-10-04 to 2026-10-11), each done by hand by a careful agent; and the README's design-record count
was kept by incrementing, not counting. This script does the ritual and leaves the judgment to the reviewer:
`archive` takes the status block as written by whoever made the call, and does everything else the same way every time.

`rank` is the view a close-out sweep starts from (the owner's `close-out` skill): the last commit that touched
each doc on ANY ref with at least WORK_LINES changed (a one-line re-point or another campaign's dated note touched
the doc; it did not work on it), plus every uncommitted doc in any worktree on this machine, flagged, because an
untracked design record is one bad afternoon from gone (docs/tasks/task-work-queue-2026-09.md, 2026-09-13).

`check` is advisory: it reports, it changes nothing, and it exits 0 unless --strict.

THE LEDGER. A doc may carry a machine-readable item ledger: the line `<!-- ledger -->` followed by a markdown table
whose first two columns are `id` and `state`. States: open, active, blocked, decision (waiting on the owner's word),
done, killed, moved (owned elsewhere; the row names where). A doc whose rows are all done, killed or moved is
closable, and `check` and `rank` say so. A doc without a ledger is reported as such, nothing more.

REFERENCES, AND WHY `--refs auto` IS THE DEFAULT. A push to main that changes only docs/**, RELEASING.md and .claude/**,
none of it read by Go code, is docs-only and goes straight through (scripts/ci_docs_only.sh, both of its rules); one
that edits a Go comment is product code, waits for CI (scripts/main_ci_guard.py), and touches files other sessions'
branches are editing. So `auto` re-points every reference in a docs-only file, and if any reference lives anywhere else
(a code comment, a test, a help string, CLAUDE.md, a generated page or a generator's input such as docs/flags.meta.json,
whose `_fresh` test would go red on a hand edit), it leaves a pointer stub at the old path, in docs/plan-still-slow.md's
shape, so those still resolve in one hop; `check` lists stubs and how many files still use each. `all` re-points
everything except generated pages, and says which regenerate commands are then owed; `stub` re-points nothing.
Frozen records are never edited: docs/completed/, docs/releases/, testdata/, CHANGELOG.md, and the records under
docs/measurements/ (everything there except its .sh and .py, which are re-pointed so a re-run still works).

Dry run by default: prints the plan and a diff. `--apply` writes the files and runs the citation lint. It never
commits; `--stage` stages exactly the paths it changed (never `git add -A`).
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import os
import pathlib
import posixpath
import re
import subprocess
import sys

ROOT = pathlib.Path(os.environ.get("TASK_DOCS_ROOT") or pathlib.Path(__file__).resolve().parent.parent).expanduser()
TASKS = "docs/tasks/"
PARKED = "docs/tasks/parked/"
COMPLETED = "docs/completed/"
HEADER_SOURCE = "docs/completed/task-cuda-cgofree-spike.md"
HEADER_START = "> **ARCHIVED — a record, not instructions.**"
FOOTER_RE = re.compile(r"\n*<!--\s*doc-reviewed:\s*(\d{4}-\d{2}-\d{2})\s*-->[ \t]*")
LEDGER_MARK = "<!-- ledger -->"
TERMINAL = {"done", "killed", "moved"}
STATES = TERMINAL | {"open", "active", "blocked", "decision"}
FROZEN_PREFIXES = ("docs/completed/", "docs/releases/", "testdata/")
FROZEN_FILES = ("CHANGELOG.md",)
LIVE_IN_MEASUREMENTS = (".sh", ".py")
DOCS_ONLY_PREFIXES = ("docs/", ".claude/")
DOCS_ONLY_FILES = ("RELEASING.md",)
LINK_RE = re.compile(r"(!?\[[^\]\n]*\]\()([^)\s]+)(\))")  # [text](target) and ![alt](target)
REFDEF_RE = re.compile(r"^(\s*\[[^\]\n]+\]:\s*)(\S+)", re.M)  # [label]: target
TICK_RE = re.compile(r"`([^`\n]+\.md(?:#[^`\n]*)?)`")  # `path/to/doc.md` in prose
GENERATED_RE = re.compile(r"(?i)generated(\*\*)? (by|from)|do not edit")  # in a doc's first lines
REGEN_RE = re.compile(r"go test [^`\n—]*?-update")
DESIGN_RE = re.compile(r"^(## Design records — `task-\*\.md` \()(\d+): (\d+) in `tasks/`, (\d+) in `tasks/parked/`\)", re.M)
ARCHIVE_RE = re.compile(r"^(## Archive — `completed/` \()(\d+)\)", re.M)


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def is_ancestor(sha: str, ref: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", sha, ref], cwd=ROOT,
                          capture_output=True).returncode == 0


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def tracked_files() -> list[str]:
    return [p for p in git("ls-files", "-z").split("\0") if p]


def is_stub(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    return len(lines) <= 12 and "**Moved.**" in text


def is_frozen(rel: str) -> bool:
    if rel in FROZEN_FILES or rel.startswith(FROZEN_PREFIXES):
        return True
    return rel.startswith("docs/measurements/") and not rel.endswith(LIVE_IN_MEASUREMENTS)


_read_cache: dict[str, bool] = {}


def read_by_go(rel: str) -> bool:
    """Rule 2 of the docs-only push, asked of scripts/ci_docs_only.sh itself: does Go code read this file? A generated
    page (its `_fresh` test reads it) and a generator's input (docs/flags.meta.json) both answer yes. Fails closed:
    anything but a clear "not-read" counts as read."""
    if rel not in _read_cache:
        sh = ROOT / "scripts" / "ci_docs_only.sh"
        if sh.exists():
            r = subprocess.run(["bash", str(sh), "--read", rel], cwd=ROOT, capture_output=True, text=True)
            _read_cache[rel] = r.stdout.strip().splitlines()[-1:] != ["not-read"]
        else:
            _read_cache[rel] = not rel.endswith(".md")
    return _read_cache[rel]


def is_docs_only(rel: str) -> bool:
    """Editing it keeps a push docs-only: under docs/ or .claude/ (or RELEASING.md), and not read by Go code."""
    return (rel in DOCS_ONLY_FILES or rel.startswith(DOCS_ONLY_PREFIXES)) and not read_by_go(rel)


def generated_by(rel: str):
    """For a generated doc (docs/option-state-grid.md, docs/kv-precision-matrix.md, ...), the command that regenerates it
    (or '' if its header names none); None for a hand-written file. A generated doc is never edited by hand: its
    `_fresh` test compares it with what the generator writes, so a hand re-point turns that test red."""
    if not rel.endswith(".md"):
        return None
    head = "\n".join(read(rel).splitlines()[:8])
    if not GENERATED_RE.search(head):
        return None
    m = REGEN_RE.search(head)
    return m.group(0).strip() if m else ""


def resolve_doc(arg: str) -> str:
    """A doc named by path or bare name -> its repo-relative path under docs/tasks/ (pointer stubs do not count)."""
    name = pathlib.Path(arg).name
    name = name if name.endswith(".md") else name + ".md"
    for rel in (f"{TASKS}{name}", f"{PARKED}{name}"):
        p = ROOT / rel
        if p.exists() and not is_stub(p.read_text(encoding="utf-8")):
            return rel
    sys.exit(f"{arg}: no live doc of that name in {TASKS} or {PARKED}")


def live_docs_on_disk() -> list[str]:
    out = []
    for d in (TASKS, PARKED):
        for p in sorted((ROOT / d).glob("*.md")):
            if not is_stub(p.read_text(encoding="utf-8")):
                out.append(p.relative_to(ROOT).as_posix())
    return out


def stubs_on_disk() -> list[str]:
    return [p.relative_to(ROOT).as_posix() for d in (TASKS, PARKED) for p in sorted((ROOT / d).glob("*.md"))
            if is_stub(p.read_text(encoding="utf-8"))]


def uncommitted(paths=("docs",)) -> dict[str, str]:
    """{path: 'untracked' | 'modified'} for .md files under the given paths."""
    out = {}
    for line in git("status", "--porcelain", "--untracked-files=all", "--", *paths).splitlines():
        code, rel = line[:2], line[3:].strip().strip('"')
        if rel.endswith(".md"):
            out[rel] = "untracked" if code == "??" else "modified"
    return out


def worktrees() -> list[pathlib.Path]:
    """Every worktree of this repo on this machine: the main checkout and the ~/goinfer-bench/*-wt ones alike."""
    out = [pathlib.Path(l[len("worktree "):]) for l in git("worktree", "list", "--porcelain").splitlines()
           if l.startswith("worktree ")]
    return out or [ROOT]


def uncommitted_everywhere(paths=("docs",)) -> list[tuple[pathlib.Path, str, str]]:
    """(worktree, path, 'untracked' | 'modified') for .md files under paths, in EVERY worktree on this machine: an
    untracked design record is at risk wherever it sits, and a sweep usually runs in a fresh worktree that has none."""
    out = []
    for wt in worktrees():
        r = subprocess.run(["git", "-C", str(wt), "status", "--porcelain", "--untracked-files=all", "--", *paths],
                           capture_output=True, text=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        for line in r.stdout.splitlines():
            code, rel = line[:2], line[3:].strip().strip('"')
            if rel.endswith(".md") and not rel.startswith("docs/measurements/"):
                out.append((wt, rel, "untracked" if code == "??" else "modified"))
    return out


def home(p: pathlib.Path) -> str:
    return str(p).replace(str(pathlib.Path.home()), "~", 1)


# ---------------------------------------------------------------------------------------------------------- ledger

def ledger(text: str):
    """Rows of the doc's `<!-- ledger -->` table as (id, state, other cells), or None if it has no ledger."""
    i = text.find(LEDGER_MARK)
    if i < 0:
        return None
    rows = []
    for line in text[i + len(LEDGER_MARK):].lstrip("\n").splitlines():
        if not line.lstrip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set("-: ") or cells[0].lower() == "id":
            continue
        state = re.sub(r"[^a-z].*$", "", cells[1].strip("*` ").lower())
        rows.append((cells[0].strip("*` "), state, cells[2:]))
    return rows


def ledger_summary(text: str) -> str:
    rows = ledger(text)
    if rows is None:
        return "no ledger"
    closed = sum(1 for _, s, _ in rows if s in TERMINAL)
    bad = [i for i, s, _ in rows if s not in STATES]
    note = f"; unknown state on {', '.join(bad)}" if bad else ""
    return f"{closed}/{len(rows)} closed{note}" + (" — CLOSABLE" if rows and closed == len(rows) else "")


# ------------------------------------------------------------------------------------------------------------ rank

WORK_LINES = 10  # a commit that changes fewer lines of a doc touched it (a re-point, another campaign's dated note)


def cmd_rank(a) -> int:
    log = git("log", "--all", "--format=%x00%H %cI", "--numstat", "--", TASKS)
    last, touched, count = {}, {}, collections.Counter()
    for chunk in log.split("\0")[1:]:  # newest first
        lines = [l for l in chunk.strip().splitlines() if l]
        sha, iso = lines[0].split(" ")
        when = dt.datetime.fromisoformat(iso)  # compare instants, not strings: offsets differ between boxes
        for row in lines[1:]:
            added, deleted, f = (row.split("\t") + ["", ""])[:3]
            if not f.endswith(".md"):
                continue
            count[f] += 1
            n = (int(added) if added.isdigit() else 0) + (int(deleted) if deleted.isdigit() else 0)
            touched.setdefault(f, when)
            # "last worked" is the newest commit that changed WORK_LINES or more; with none, the newest touch
            if f not in last or (last[f][2] < WORK_LINES <= n):
                last[f] = (when, sha, n)
    dirty = {(wt, rel): st for wt, rel, st in uncommitted_everywhere((TASKS,))}
    # docs some unmerged ref already archives: another session is closing it; a sweep leaves it alone
    elsewhere = set(git("log", "--all", "--not", a.main, "--format=", "--name-only", "--", COMPLETED).split())
    rows = []
    on_ref = [p for p in git("ls-tree", "-r", "--name-only", a.main, "--", TASKS).split() if p.endswith(".md")]
    texts = {p: git("show", f"{a.main}:{p}") for p in on_ref}
    for (wt, rel), st in dirty.items():
        if st != "untracked" or rel in on_ref:
            continue
        f = wt / rel
        key = f"{rel}  (only in {home(wt)})"
        texts[key] = f.read_text(encoding="utf-8")
        stat = f.stat()
        when = dt.datetime.fromtimestamp(stat.st_mtime).astimezone()
        born = dt.datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_mtime)).date().isoformat()
        rows.append((when, key, born, 0, ["UNCOMMITTED: never committed on any ref"]))
    modified = {rel for (wt, rel), st in dirty.items() if st == "modified"}
    for rel in [p for p in on_ref if not is_stub(texts[p])]:
        flags = []
        if COMPLETED + pathlib.Path(rel).name in elsewhere:
            flags.append("ARCHIVED ON AN UNMERGED REF (another session is closing it)")
        if rel not in last:
            continue
        when, sha, _ = last[rel]
        if touched[rel] > when:
            flags.append(f"touched {touched[rel].astimezone():%Y-%m-%d}")
        if rel in modified:
            flags.append("uncommitted edits in a worktree")
        if not is_ancestor(sha, a.main):
            flags.append(f"latest work not on {a.main}")
        rows.append((when, rel, None, count[rel], flags))
    rows.sort(key=lambda r: r[0], reverse=True)
    if a.top:
        rows = rows[: a.top]
    print(f"{'#':>2}  {'last worked':16}  {'created':10}  {'commits':>7}  {'lines':>5}  {'reviewed':10}  doc  [ledger; flags]")
    for i, (when, rel, born, n, flags) in enumerate(rows, 1):
        text = texts[rel]
        if born is None:
            born = (git("log", "--all", "--follow", "--diff-filter=A", "--format=%cs", "--", rel).split() or ["?"])[-1]
        foot = FOOTER_RE.findall(text)
        extra = "; ".join([ledger_summary(text), *flags])
        print(f"{i:>2}  {when.astimezone():%Y-%m-%d %H:%M}  {born:10}  {n:>7}  {len(text.splitlines()):>5}  "
              f"{(foot[-1] if foot else 'none'):10}  {rel.removeprefix('docs/')}  [{extra}]")
    return 0


# ----------------------------------------------------------------------------------------------------------- check

def actual_counts() -> tuple[int, int, int]:
    in_tasks = sum(1 for p in (ROOT / TASKS).glob("task-*.md") if not is_stub(p.read_text(encoding="utf-8")))
    in_parked = sum(1 for p in (ROOT / PARKED).glob("task-*.md") if not is_stub(p.read_text(encoding="utf-8")))
    return in_tasks + in_parked, in_tasks, in_parked


def archive_count() -> int:
    return len(list((ROOT / COMPLETED).iterdir()))  # the skill's `ls docs/completed | wc -l`


def recount_readme(text: str) -> str:
    total, in_tasks, in_parked = actual_counts()
    text = DESIGN_RE.sub(lambda m: f"{m.group(1)}{total}: {in_tasks} in `tasks/`, {in_parked} in `tasks/parked/`)", text)
    return ARCHIVE_RE.sub(lambda m: f"{m.group(1)}{archive_count()})", text)


def references(rel: str) -> list[str]:
    """Tracked text files, frozen records excluded, that point at rel by repo path or by a relative link."""
    out = []
    for f in git("grep", "-l", "-I", "-F", pathlib.Path(rel).name, "--", ".", check=False).splitlines():
        if f != rel and not is_frozen(f) and rewrite_refs(read(f), f, rel, rel + ".probe") is not None:
            out.append(f)
    return out


def cmd_check(a) -> int:
    problems = 0
    dirty = uncommitted_everywhere(("docs",))
    print("Uncommitted docs in this repo's worktrees (an untracked design record can be lost; commit it or say why not):")
    for wt, rel, st in sorted(dirty, key=lambda t: (t[2], t[1])):
        print(f"  {st:9}  {rel}  ({home(wt)})")
        problems += st == "untracked"
    if not dirty:
        print("  none")

    files = tracked_files()
    missing = [p for p in files if p.startswith(COMPLETED) and p.endswith(".md")
               and HEADER_START not in "\n".join(read(p).splitlines()[:40])]
    print(f"\nArchived docs without the ARCHIVED header ({HEADER_SOURCE} has it): {len(missing)}")
    for p in missing:
        print(f"  {p}")
    problems += len(missing)

    live = live_docs_on_disk()
    nofoot = [p for p in live if not FOOTER_RE.search(read(p))]
    print(f"\nLive docs without a <!-- doc-reviewed: --> footer: {len(nofoot)} of {len(live)}")
    for p in nofoot if a.verbose else []:
        print(f"  {p}")

    print("\nLedgers:")
    without = 0
    for p in live:
        rows = ledger(read(p))
        if rows is None:
            without += 1
        else:
            print(f"  {ledger_summary(read(p)):34}  {p}")
    print(f"  {without} of {len(live)} live docs have no ledger")

    text = read("docs/README.md")
    print("\ndocs/README.md counts (recounted from the tree; pointer stubs are not design records):")
    m = DESIGN_RE.search(text)
    if m:
        claimed, actual = tuple(map(int, m.groups()[1:])), actual_counts()
        problems += claimed != actual
        print(f"  design records: README says {claimed[0]} ({claimed[1]} + {claimed[2]}), tree has {actual[0]} "
              f"({actual[1]} + {actual[2]}), plus {len(stubs_on_disk())} pointer stubs" + ("" if claimed == actual else "  <- recount"))
    m = ARCHIVE_RE.search(text)
    if m:
        problems += int(m.group(2)) != archive_count()
        print(f"  archive: README says {m.group(2)}, tree has {archive_count()}" + ("" if int(m.group(2)) == archive_count() else "  <- recount"))

    print("\nPointer stubs, and how many live files still use the old path:")
    for rel in stubs_on_disk():
        users = references(rel)
        print(f"  {len(users):3} files  {rel}" + ("  <- nothing uses it; the stub can go" if not users else ""))

    print("\nLive references to a docs/tasks/ path that no longer exists:")
    dangling = 0
    pat = re.compile(r"docs/tasks/(?:parked/)?[A-Za-z0-9._-]+\.md")
    for f in git("grep", "-l", "-I", "-E", r"docs/tasks/", "--", ".", check=False).splitlines():
        if is_frozen(f):
            continue
        body = read(f)
        for mm in pat.finditer(body):
            if not (ROOT / mm.group(0)).exists():
                dangling += 1
                print(f"  {f}:{body.count(chr(10), 0, mm.start()) + 1}  {mm.group(0)}")
    print("  none" if not dangling else f"  ({dangling})")
    return 1 if (a.strict and problems) else 0


# -------------------------------------------------------------------------------------------------------- evidence

ID_RE = re.compile(r"(?<![A-Za-z0-9-])([A-Z]{1,4}(?:-[A-Z]{1,4})?-?\d{1,2}[a-z]?)(?![A-Za-z0-9])")  # KV8a, MEM17, G-W3b, G-CL1


def prefix(t: str) -> str:
    return re.sub(r"-?\d.*$", "", t)


def doc_ids(text: str) -> list[str]:
    """Item IDs the doc owns: tokens like KV4, MEM17, S6, G-W2, KV8a whose prefix appears with two or more numbers in
    the doc's headings, table first cells or bold line starts. A hint list, not a verdict."""
    anchors = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.count("|") > 1:
            anchors += ID_RE.findall(s.split("|")[1])
        elif s.startswith(("#", "**", "- **")):
            anchors += ID_RE.findall(s[:60])
    nums = collections.defaultdict(set)
    for t in anchors:
        nums[prefix(t)].add(re.search(r"\d+", t).group())
    own = {p for p, n in nums.items() if len(n) >= 2}
    seen = list(dict.fromkeys(t for t in ID_RE.findall(text) if prefix(t) in own))
    return sorted(seen, key=lambda t: (prefix(t), int(re.search(r"\d+", t).group()), t))


def cmd_evidence(a) -> int:
    rel = resolve_doc(a.doc)
    stem = pathlib.Path(rel).stem
    slug = re.sub(r"-\d{4}-\d{2}(-\d{2})?$", "", stem.removeprefix("task-"))
    ids = doc_ids(read(rel))
    fams = collections.Counter(prefix(t) for t in ids)
    main_fam = next((p for p, _ in fams.most_common() if len(p) >= 2), None)
    ctx = [re.escape(stem), re.escape(slug), re.escape(slug.replace("-", " "))]
    if main_fam:
        ctx.append(rf"(?<![A-Za-z0-9]){re.escape(main_fam)}\d")
    ctx_re = re.compile("|".join(ctx), re.I)
    print(f"{rel}: {len(ids)} item IDs ({', '.join(ids)})")
    print(f"A commit counts as this doc's when it names an ID of a two-letter-or-longer family, or names an ID and also the\n"
          f"doc ('{slug}'{f' or any {main_fam}n' if main_fam else ''}). Newest first; a hint for the reviewer, not a verdict.\n")
    for t in ids:
        pat = rf"(^|[^A-Za-z0-9-]){re.escape(t)}([^A-Za-z0-9]|$)"
        entries = [e.split("\x1f", 1) for e in git("log", a.main, "-E", f"--grep={pat}", "--format=%h %cs %s\x1f%b\x1e")
                   .split("\x1e") if e.strip()]
        mine = [s.strip() for s, b in entries if len(prefix(t)) >= 2 or ctx_re.search(s + b)]
        branch = git("log", "--all", "--not", a.main, "-E", f"--grep={pat}", "--format=%h").split()
        print(f"{t:8} {len(mine):3} this doc's" + (f" (of {len(entries)} naming {t})" if len(entries) != len(mine) else "")
              + (f", {len(branch)} on unmerged refs" if branch else ""))
        for s in mine[: a.n]:
            print(f"         {s[:150]}")
    return 0


# --------------------------------------------------------------------------------------------------------- archive

def canonical_header() -> str:
    lines = read(HEADER_SOURCE).splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(HEADER_START))
    end = start
    while end < len(lines) and lines[end].startswith(">"):
        end += 1
    return "\n".join(lines[start:end])


def _retarget(target: str, from_file: str, old: str, new: str):
    """If target (as written in from_file) points at old, the equivalent target pointing at new; else None."""
    if re.match(r"^[a-z]+:|^#|^/", target):
        return None
    path, _, frag = target.partition("#")
    if not path:
        return None
    suffix = f"#{frag}" if frag else ""
    if path == old:  # a repo-root path
        return new + suffix
    base = posixpath.dirname(from_file)
    if posixpath.normpath(posixpath.join(base, path)) == old:
        return posixpath.relpath(new, base or ".") + suffix
    return None


def rewrite_refs(text: str, from_file: str, old: str, new: str):
    """text with every reference to old re-pointed at new, or None if it has none."""
    def link(m):
        t = _retarget(m.group(2), from_file, old, new)
        return m.group(0) if t is None else m.group(1) + t + m.group(3)

    def refdef(m):
        t = _retarget(m.group(2), from_file, old, new)
        return m.group(0) if t is None else m.group(1) + t

    def tick(m):
        t = _retarget(m.group(1), from_file, old, new)
        return m.group(0) if t is None else f"`{t}`"

    out = TICK_RE.sub(tick, REFDEF_RE.sub(refdef, LINK_RE.sub(link, text)))
    # repo-root paths anywhere else: code comments, scripts, prose, ../docs/... reads, github.com/.../blob/main/docs/...
    out = re.sub(rf"(?<![A-Za-z0-9_.-]){re.escape(old)}", new, out)
    return out if out != text else None


def move_outbound_links(text: str, old: str, new: str) -> str:
    """The doc's own relative links, rewritten so they still resolve from its new directory."""
    old_dir, new_dir = posixpath.dirname(old), posixpath.dirname(new)

    def fix(target: str) -> str:
        if re.match(r"^[a-z]+:|^#|^/", target):
            return target
        path, _, frag = target.partition("#")
        if not path or path.startswith(("docs/", "./docs/")):
            return target
        moved = posixpath.relpath(posixpath.normpath(posixpath.join(old_dir, path)), new_dir)
        return moved + (f"#{frag}" if frag else "")

    text = LINK_RE.sub(lambda m: m.group(1) + fix(m.group(2)) + m.group(3), text)
    return REFDEF_RE.sub(lambda m: m.group(1) + fix(m.group(2)), text)


def archived_text(text: str, status: str, header: str) -> str:
    lines = text.splitlines()
    title, body = (lines[0], "\n".join(lines[1:]).lstrip("\n")) if lines and lines[0].startswith("# ") else (None, text)
    body = FOOTER_RE.sub("", body).rstrip() + "\n"  # docs/completed/ files carry no review footer
    status = status.strip()
    if not status.startswith(">"):
        status = "\n".join("> " + l if l else ">" for l in status.splitlines())
    return "\n".join(([title, ""] if title else []) + [header, "", status, "", body])


def stub_text(text: str, old: str, new: str, line: str) -> str:
    title = text.splitlines()[0] if text.startswith("# ") else f"# {pathlib.Path(old).stem}"
    link = posixpath.relpath(new, posixpath.dirname(old))
    what = f" {line.strip()}" if line else ""
    return (f"{title}\n\n**Moved.** Archived {dt.date.today().isoformat()}; the doc now lives at\n"
            f"[`{new.removeprefix('docs/')}`]({link}).{what} Pointer kept here because other pages and code comments\n"
            f"link to this path.\n")


def cmd_archive(a) -> int:
    old = resolve_doc(a.doc)
    if old in uncommitted((TASKS,)):
        sys.exit(f"{old} has uncommitted changes; commit them first, so the archive commit is only the archive")
    new = COMPLETED + pathlib.Path(old).name
    if (ROOT / new).exists():
        sys.exit(f"{new} already exists")
    status = sys.stdin.read() if a.status == "-" else pathlib.Path(a.status).read_text(encoding="utf-8")
    if not status.strip():
        sys.exit("the status block is empty: say what closed the doc, what the tree says now, and what it did not own")

    refs = references(old)
    generated = {f: generated_by(f) for f in refs if generated_by(f) is not None}
    code_side = [f for f in refs if not is_docs_only(f) or f in generated]
    mode = ("docs+stub" if code_side else "docs") if a.refs == "auto" else a.refs
    repoint = [f for f in refs if f not in generated and
               (mode == "all" or (mode in ("docs", "docs+stub") and is_docs_only(f)))]
    keep_stub = mode in ("stub", "docs+stub") or bool(generated)  # a generated doc keeps pointing at the old path

    text = read(old)
    planned = {new: archived_text(move_outbound_links(text, old, new), status, canonical_header())}
    for f in repoint:
        out = rewrite_refs(read(f), f, old, new)
        if out is not None:
            planned[f] = out
    if keep_stub:
        planned[old] = stub_text(text, old, new, a.stub_line or "")

    print(f"archive {old} -> {new}")
    print(f"  live references (frozen records excluded): {len(refs)} file(s), {len(code_side)} outside docs/ and .claude/")
    print(f"  mode {mode}: re-point {len(repoint)} file(s)" + ("; leave a pointer stub at the old path" if keep_stub else ""))
    for f in sorted(set(refs) - set(repoint)):
        why = (f"  (generated: re-point its generator, then `{generated[f] or 'its regenerate command'}`)"
               if f in generated else "")
        print(f"    still pointing at the stub: {f}{why}")
    if generated and mode == "all":
        print("  REQUIRED before committing: the generators were re-pointed, so regenerate: "
              + "; ".join(sorted({c for c in generated.values() if c})))
    if not a.apply:
        for f, out in planned.items():
            before = text if f in (new, old) else read(f)
            sys.stdout.writelines(difflib.unified_diff(before.splitlines(True), out.splitlines(True),
                                                       f"a/{old if f == new else f}", f"b/{f}", n=1))
        print("\n(dry run: nothing written; add --apply)")
        return 0

    subprocess.run(["git", "mv", old, new], cwd=ROOT, check=True)
    for f, out in planned.items():
        (ROOT / f).write_text(out, encoding="utf-8")
    readme = read("docs/README.md")
    (ROOT / "docs/README.md").write_text(recount_readme(readme), encoding="utf-8")
    changed = sorted(set(planned) | {"docs/README.md"} | ({old} if keep_stub else set()))
    lint = subprocess.run([sys.executable, "scripts/queue_citation_lint.py"], cwd=ROOT)
    print(f"\ncitation lint exit {lint.returncode}" + ("" if lint.returncode == 0 else
          " — read it; rows keyed to the moved doc are refreshed by `--update`, never by deleting a citation"))
    if a.stage:
        subprocess.run(["git", "add", "--", *changed], cwd=ROOT, check=True)
        print("staged (with the rename): " + " ".join(changed))
    else:
        print("the rename is staged; stage the rest with: git add -- " + " ".join(changed))
    print(f"suggested subject: docs: archive {pathlib.Path(old).stem} (<what closed it>); its path updated in "
          f"{len(repoint)} file(s)" + ("; a pointer stub stays at the old path" if keep_stub else ""))
    return lint.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--main", default="origin/main", help="the ref that counts as shipped (default origin/main)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="live task docs, most recently created or worked first")
    r.add_argument("--top", type=int, default=0)
    c = sub.add_parser("check", help="advisory hygiene report; changes nothing")
    c.add_argument("--strict", action="store_true", help="exit 1 on uncommitted docs, missing headers or bad counts")
    c.add_argument("-v", "--verbose", action="store_true", help="list the live docs without a footer")
    e = sub.add_parser("evidence", help="the commits that name each of a doc's item IDs")
    e.add_argument("doc")
    e.add_argument("-n", type=int, default=3, help="commits shown per ID (default 3)")
    ar = sub.add_parser("archive", help="the doc-review §3 archive procedure, mechanically (dry run unless --apply)")
    ar.add_argument("doc")
    ar.add_argument("--status", required=True, help="file holding the dated status block ('-' reads stdin)")
    ar.add_argument("--refs", choices=("auto", "docs", "all", "stub"), default="auto")
    ar.add_argument("--stub-line", help="one sentence for the pointer stub: what became of the work")
    ar.add_argument("--apply", action="store_true", help="write the files and run the citation lint")
    ar.add_argument("--stage", action="store_true", help="git add exactly the changed paths")
    a = p.parse_args()
    return {"rank": cmd_rank, "check": cmd_check, "evidence": cmd_evidence, "archive": cmd_archive}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
