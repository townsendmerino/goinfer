#!/usr/bin/env python3
"""comment_census.py: CC6 of docs/tasks/task-code-comments-2026-10.md. Keeps Go comments from growing back into a lab notebook.

  comment_census.py report [--tests] [PATH...]
      The §1 census per package: files, non-blank lines, comment lines (and their share), comment blocks of 1-3 lines and how many carry
      no history marker, blocks of 10+ lines and how many carry one, and exported top-level doc comments in public packages that carry
      one. Test files are left out unless --tests (the task's own figures are "outside _test.go"). PATHs are directories or files
      (default: every tracked .go file). Comments inside raw strings (embedded kernel text) and `//go:` style directives are not
      comments.

  comment_census.py diff [--base REV] [--head REV] [--strict]
      The pre-push warning. Prints every comment added or changed between REV and HEAD in a .go file that carries a history marker: a
      date, a commit id, a tracker id (M-35, N-79), a figure with a unit, or a `file.go:NNN` reference. Default base: the merge base of
      HEAD with origin/main. It WARNS: the exit status is 0 whatever it finds, unless --strict. A marker is fine on a pointer
      ("docs/code-notes/gpu.md#runModelToModelW, M-08") and "timeout in ms" is a contract, not history, so read each hit; the rule is
      CLAUDE.md § "Code comments". Directives, raw strings, comment text the diff did not add, and a marker the file's comments already carried at the base (a kept label, a
      rewrapped line) are never reported.

The report's marker is the broad one the task's §1 census defined (it also counts docs/ and task- references and phrases such as "used
to" and "landed"); the diff warning uses the narrow five-kind marker, because a broad one would fire on every pointer."""
import argparse
import collections
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comment_diet as cd  # noqa: E402

# the §1 census marker: a date, a commit id, an audit or tracker id, a docs/ or task- reference, a measured figure with a unit, or a
# history phrase. comment_diet.MARK is that regex.
BROAD = cd.MARK

DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
# a commit id: 7-12 hex characters holding at least one digit and one letter (so "deadbeef" counts, "1234567" and "defaced" do not)
COMMIT = re.compile(r"(?<![\w./-])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,12}(?![\w./-])")
TRACKER = re.compile(r"(?<![\w-])[A-Z]{1,2}-\d{1,3}[a-z]?(?![\w-])")
FIGURE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s?(?:ms|µs|us|ns|tok/s|GB|MB|KB|GiB|MiB|×)(?![\w])|(?<![\w.])\d+(?:\.\d+)?x(?![\w])")
FILELINE = re.compile(r"\b[\w/]+\.(?:go|py|sh):\d+")
KINDS = (("date", DATE), ("commit id", COMMIT), ("tracker id", TRACKER), ("figure with a unit", FIGURE), ("file.go:NNN", FILELINE))
# an id that is plainly not a tracker: ISO-ish names and well-known hyphenated tokens
NOT_TRACKER = {"UTF-8", "SHA-1", "SHA-2", "X-1", "A-Z", "AES-128", "AES-256"}


# a path or a doc pointer (docs/measurements/x-2026-09-23.md, decoder/forwardn.go, notes.md#Heading): the date or id inside a file name
# is a name, not history, and a tracker id on the same line as a doc pointer is a label on that pointer
PATHLIKE = re.compile(r"[\w.#~-]*(?:/|\.md\b)[\w./#~-]*")


def markers(text: str):
    """The marker kinds `text` carries (the narrow, five-kind set). File names and pointers are blanked first, and a tracker id on a line
    that carries a doc pointer is a label on it, not a finding."""
    pointer = "docs/" in text or ".md" in text
    flat = PATHLIKE.sub(lambda m: m.group(0) if FILELINE.fullmatch(m.group(0)) else " ", text)
    out = []
    for kind, rx in KINDS:
        if kind == "tracker id" and pointer:
            continue
        for m in rx.finditer(flat):
            if kind == "tracker id" and m.group(0) in NOT_TRACKER:
                continue
            out.append((kind, m.group(0)))
    return out


def git(*args, cwd=None):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd).stdout


def root():
    return git("rev-parse", "--show-toplevel").strip()


# ---------------------------------------------------------------- report

PUBLIC_EXCLUDE = ("internal/", "cmd/", "demo/", "examples/", "site/", "scripts/", "testdata/")
EXPORTED_DECL = re.compile(r"^(?:func\s+(?:\([^)]*\)\s*)?|type\s+|var\s+|const\s+)([A-Z]\w*)")


def is_public(pkg: str) -> bool:
    return not any(pkg + "/" == x or (pkg + "/").startswith(x) or ("/" + x) in ("/" + pkg + "/") for x in PUBLIC_EXCLUDE)


def census_file(path: str):
    src = open(path, errors="replace").read()
    lines, bl = cd.blocks(src)
    nonblank = sum(1 for l in lines if l.strip())
    row = dict(nonblank=nonblank, comment=0, short=0, short_unmarked=0, long=0, long_lines=0, long_marked=0, doc=0, doc_marked=0)
    for a, b in bl:
        n = b - a + 1
        text = " ".join(lines[a - 1:b])
        marked = bool(BROAD.search(text))
        row["comment"] += n
        if n <= 3:
            row["short"] += 1
            row["short_unmarked"] += 0 if marked else 1
        if n >= 10:
            row["long"] += 1
            row["long_lines"] += n
            row["long_marked"] += 1 if marked else 0
        # an exported top-level doc comment: the block ends directly above an exported declaration at column 0
        if b < len(lines) and EXPORTED_DECL.match(lines[b]):
            row["doc"] += 1
            row["doc_marked"] += 1 if marked else 0
    return row


def cmd_report(args):
    r = root()
    files = [f for f in git("ls-files", "*.go", cwd=r).split() if args.tests or not f.endswith("_test.go")]
    if args.paths:
        want = [os.path.relpath(os.path.abspath(p), r) for p in args.paths]
        files = [f for f in files if any(f == w or f.startswith(w.rstrip("/") + "/") for w in want)]
    by = collections.defaultdict(lambda: collections.Counter())
    for f in files:
        pkg = os.path.dirname(f) or "."
        row = census_file(os.path.join(r, f))
        c = by[pkg]
        c["files"] += 1
        for k, v in row.items():
            c[k] += v
    cols = ("package", "files", "lines", "comment", "%", "1-3", "unmk", "10+", "10+ln", "10+mk", "pubdoc", "pubdoc-mk")
    fmt = "{:30} {:>5} {:>7} {:>7} {:>4} {:>5} {:>5} {:>4} {:>6} {:>5} {:>6} {:>9}"
    print(fmt.format(*cols))
    tot = collections.Counter()
    for pkg in sorted(by, key=lambda p: -by[p]["comment"]):
        c = by[pkg]
        pub = is_public(pkg)
        print(fmt.format(pkg[:30], c["files"], c["nonblank"], c["comment"], f"{100 * c['comment'] // max(c['nonblank'], 1)}", c["short"], c["short_unmarked"],
                         c["long"], c["long_lines"], c["long_marked"], c["doc"] if pub else "-", c["doc_marked"] if pub else "-"))
        tot.update(c)
        if pub:
            tot["pubdoc"] += c["doc"]
            tot["pubdoc_marked"] += c["doc_marked"]
    print(fmt.format("TOTAL", tot["files"], tot["nonblank"], tot["comment"], f"{100 * tot['comment'] // max(tot['nonblank'], 1)}", tot["short"], tot["short_unmarked"],
                     tot["long"], tot["long_lines"], tot["long_marked"], tot["pubdoc"], tot["pubdoc_marked"]))
    print("\ncolumns: lines = non-blank lines; comment = whole-line // comment lines; 1-3 = comment blocks of 1-3 lines, unmk = of those with no history "
          "marker; 10+ = blocks of 10+ lines, 10+ln their lines, 10+mk those carrying a marker; pubdoc = exported top-level doc comments in public "
          "packages, pubdoc-mk those carrying a marker. Marker = the task's §1 definition (date, commit id, tracker id, docs/ or task- reference, "
          "figure with a unit, history phrase)." + ("" if args.tests else " Test files excluded (--tests includes them)."))
    return 0


# ---------------------------------------------------------------- diff

ZERO = "0" * 40


def added_lines(base: str, head: str):
    """{file -> set of new-file line numbers added or changed between base and head} for tracked .go files."""
    out = collections.defaultdict(set)
    diff = git("diff", "-U0", "--no-renames", "--diff-filter=AM", base, head, "--", "*.go")
    cur = None
    for ln in diff.split("\n"):
        if ln.startswith("+++ b/"):
            cur = ln[6:]
        elif ln.startswith("@@") and cur:
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", ln)
            if m:
                start, cnt = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
                out[cur].update(range(start, start + cnt))
    return out


def file_at(head: str, path: str) -> str:
    if head in ("", "WORKTREE"):
        return open(path, errors="replace").read()
    return git("show", f"{head}:{path}")


def cmd_diff(args):
    base = args.base
    if not base:
        base = git("merge-base", "HEAD", "origin/main").strip() or "HEAD~1"
    head = args.head or "HEAD"
    hits = []
    for path, nums in sorted(added_lines(base, head).items()):
        src = file_at(head, path)
        comments = cd.line_comments(src)
        # a marker the file's comments already carried at the base is a label that was kept or a line that was rewrapped, not history added now
        had = {tok for text in cd.line_comments(git("show", f"{base}:{path}")).values() for _, tok in markers(text)}
        for n in sorted(nums):
            text = comments.get(n)
            if text is None or text.lstrip().startswith(cd.DIRECTIVE):
                continue
            ms = [(k, tok) for k, tok in markers(text) if tok not in had]
            if ms:
                hits.append((path, n, text.strip(), ms))
    if not hits:
        print(f"comment_census: no history markers in the comments added since {base[:10]}")
        return 0
    print(f"comment_census: WARNING: {len(hits)} comment line(s) added or changed since {base[:10]} carry a history marker "
          f"(CLAUDE.md § \"Code comments\": history goes to the commit message, the measurement or task doc, or docs/code-notes; a marker is fine on a pointer):")
    for path, n, text, ms in hits[:60]:
        kinds = ", ".join(sorted({k for k, _ in ms}))
        print(f"  {path}:{n}: [{kinds}] {text[:150]}")
    if len(hits) > 60:
        print(f"  ... and {len(hits) - 60} more")
    return 1 if args.strict else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--tests", action="store_true", help="include _test.go files")
    r.add_argument("paths", nargs="*")
    d = sub.add_parser("diff")
    d.add_argument("--base", default="")
    d.add_argument("--head", default="")
    d.add_argument("--strict", action="store_true", help="exit 1 when anything is found (the warning exits 0)")
    args = ap.parse_args(argv)
    return cmd_report(args) if args.cmd == "report" else cmd_diff(args)


if __name__ == "__main__":
    sys.exit(main())
