#!/usr/bin/env python3
"""One-off (CC0 step 2, docs/tasks/task-code-comments-2026-10.md): rewrite every `path:line` in a LIVE doc to `path:Name`.

    python3 scripts/migrate_citations_to_symbols.py [--dry-run] [--out DIR] [DOC ...]

HOW A CITATION IS RESOLVED (and why the stale number is NOT trusted). The old lint recorded the CONTENT of every cited line in docs/citation-index.md,
and when that content moved it accepted the move while leaving the stale number in the prose. A converter that took the number at face value named
whatever function now occupies that old line (found 2026-10-10: a cancellation-loop sentence naming warnPrefillDeclined and normalize; three
registry.go architectures replaced by olmo3, olmoHybrid and lfm2). So each first number is located by its RECORDED CONTENT: still at the cited line ->
that line; elsewhere, exactly once -> there; a recorded `anchor: <declaration>` -> that declaration. A citation the index holds no content for (the
UNKEYABLE ones, and every later number in a comma list, which the lint never read) is converted only when the declaration it lands in is NAMED IN THE
SURROUNDING PROSE; otherwise it goes on the hand list. drift-list.tsv records every citation whose recorded-content resolution differs from the
naive one, so the size of the problem is on the page.

What it does. For each unpinned live doc (every tracked markdown doc the lint reads, minus the dated records below, minus any doc that
already carries a `citations-at` marker) it finds each `path:line` (and `path:line-end`), resolves the file through the lint's own roots
(the module cache at go.mod's version for aikit, then the checkouts), asks the lint's own scanner which declaration encloses that line, and
rewrites the citation to `path:Name` (`path:Type.Method` for a method). The scanner is the lint's (queue_citation_lint.go_decls), so what
this writes is exactly what the lint will check; the two cannot disagree about what a declaration is.

What it does NOT do. It never inserts a quoted fragment — whether the prose depends on one specific statement is a reading task, and a
guessed fragment is worse than none. Instead it writes two lists to --out:

  review-list.tsv  every citation that lands inside a declaration longer than REVIEW_LINES lines: doc, old cite, new cite, the declaration's
                   length, and the cited line's text. A reader decides, per row, whether the sentence needs a fragment.
  hand-list.tsv    every citation it left alone and why: outside any declaration (a package clause, an import block, a file-level comment, a
                   blank between declarations), a name that is not unique in its file (`init`), or a file it cannot resolve. These stay
                   `path:line` until someone converts them by hand.

IDEMPOTENT. A converted citation is no longer a `path:line`, so a second run finds nothing to do for it; re-run it after rebasing onto main
(main gains new `path:line` citations daily) and it converts only those. The lists are MERGED across runs (keyed on doc, old cite, new
cite), never overwritten, so the first run's long-function rows survive the re-runs that follow.

Also strips the `#L123` / `#L10-L20` line anchors from markdown link targets in the docs it converts: they are line numbers in the same way,
are invisible to the lint, and had already come apart from their own labels (how-inference-works.md: label 1755-1782, anchor #L1185-L1532).
"""
import argparse
import collections
import importlib.util
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("queue_citation_lint", HERE / "queue_citation_lint.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

REVIEW_LINES = 40

# The dated records, judged 2026-10-10 (CC0 step 3 pins these; they are NOT migrated). A record describes code at a point in time and is
# not maintained as current state: an audit report, a measurement write-up or pre-registration, review notes for a drafted writeup. Once
# a doc carries `citations-at` this list no longer matters to it; it exists so step 2 can run before step 3. Task docs, queues, reference
# docs and policies are LIVE even when their name says "audit" (task-recompute-audit.md and task-verification-surface-audit.md carry
# per-item status columns and dated status corrections: they are maintained, so they are converted).
RECORD_PREFIXES = ("docs/measurements/", "site/different/REVIEW-")
RECORD_DOCS = {"docs/audit-2026-09-10.md", "docs/audit-metal-2026-09-12.md", "docs/audit-metal-2026-09-30.md"}


def is_record(rel: str) -> bool:
    return rel in RECORD_DOCS or rel.startswith(RECORD_PREFIXES)


# A line citation as the docs actually write it: `path:12`, `path:12-20`, `path:12–20` (an EN DASH: 28 of them), and `path:12,34,56` (a
# comma list of more lines in the same file: 35 of them). The first migration run read only the first number and left the rest behind as
# `Name,34,56` and `Name–20`, which the lint cannot see; so a list or a range is converted as ONE unit.
CITE_RE = re.compile(r"(?<![\w/.])((?:[a-z0-9_]+/)*[a-z0-9_]+\.(?:go|sh|py)):(\d+)(?:[-–](\d+))?((?:,\d+(?:[-–]\d+)?)*)(?![\d])")
EXTRA_RE = re.compile(r",(\d+)(?:[-–](\d+))?")
ANCHOR_RE = re.compile(r"(\]\([^)\s]*\.(?:go|py|sh))#L\d+(?:-L\d+)?\)")


def py_enclosing(lines, line):
    """(name, start, end) of the def/class enclosing a 1-based line of a Python file, `Class.method` for a method; None outside any."""
    decl = re.compile(r"^(\s*)(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)")
    chain = []  # innermost first
    indent_limit = None
    for i in range(min(line, len(lines)) - 1, -1, -1):
        m = decl.match(lines[i])
        if not m:
            continue
        ind = len(m.group(1))
        if indent_limit is not None and ind >= indent_limit:
            continue
        # the cited line must lie inside this def/class's body: every non-blank line between must be deeper than its indent
        if i + 1 < line and any(lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= ind for j in range(i + 1, line)):
            continue
        chain.append((m.group(2), m.group(3), ind, i + 1))
        indent_limit = ind
        if ind == 0:
            break
    if not chain:
        return None
    name = ".".join(c[1] for c in reversed(chain) if True)
    # `Class.method` only for a def directly under a class; deeper nesting is not citable, so name the outer def
    kinds = [c[0] for c in reversed(chain)]
    if len(chain) >= 2 and kinds[0] == "class" and kinds[1] == "def":
        name = f"{chain[-1][1]}.{chain[-2][1]}"
    else:
        name = chain[-1][1]
    start = chain[0][3]
    end = start
    ind0 = chain[0][2]
    for j in range(start, len(lines)):
        if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= ind0:
            break
        end = j + 1
    return name, start, end


_INDEX = None


def load_index():
    """{`doc|path:line` -> recorded content} from docs/citation-index.md as it stands (run BEFORE `--update` rewrites it); `UNKEYABLE` stays as is."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        ip = q.index_path()
        if ip.exists():
            t = ip.read_text()
            if q.MARK_BEGIN in t:
                for ln in t.split(q.MARK_BEGIN, 1)[1].splitlines():
                    m = re.match(r"\|\s*`([^`]+:\d+)`\s*\|\s*([^|]*?)\s*\|\s*`(.*)`\s*\|$", ln.strip())
                    if m:
                        _INDEX[m.group(1)] = m.group(3).replace("\\|", "|")
    return _INDEX


def _prose_names(name, prose):
    """True when the surrounding prose names the declaration (its last component, as a whole word)."""
    tail = name.split(".")[-1]
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(tail) + r"(?![A-Za-z0-9_])", prose))


def locate(doc, rel, line, first, lines):
    """(resolved_line, how) or (None, reason). how is cited | moved | anchor | unkeyed."""
    rec = load_index().get(f"{doc}|{rel}:{line}") if first else None
    if rec and rec != "UNKEYABLE":
        if rec.startswith("anchor: "):
            decl = rec[len("anchor: "):]
            hits = [i + 1 for i, l in enumerate(lines) if l.strip()[:80] == decl]
            if len(hits) == 1:
                return hits[0], "anchor"
            return None, f"its recorded anchor `{decl[:50]}` is {'gone' if not hits else 'now at ' + str(len(hits)) + ' places'}"
        if 1 <= line <= len(lines) and lines[line - 1].strip()[:88] == rec:
            return line, "cited"
        hits = [i + 1 for i, l in enumerate(lines) if l.strip()[:88] == rec]
        if len(hits) == 1:
            return hits[0], "moved"
        return None, f"its recorded content `{rec[:50]}` is {'gone' if not hits else 'now at ' + str(len(hits)) + ' places'}"
    return line, "unkeyed"


def classify(rel, line, doc=None, first=True, prose=""):
    """('ok', name, decl_len, cited_text, how, naive_name) | ('hand', reason, None, cited_text, how, None)."""
    _repo, base = q.resolve_base(rel)
    if base is None:
        return "hand", "the file resolves in no repository here", None, "", "", None
    lines = (base / rel).read_text(errors="replace").split("\n")
    if not 1 <= line <= len(lines):
        return "hand", f"line {line} is past the end of the file ({len(lines)} lines)", None, "", "", None
    at, how = locate(doc, rel, line, first, lines)
    if at is None:
        return "hand", how, None, lines[line - 1].strip(), "", None
    cited = lines[at - 1].strip()

    def decl_at(n):
        if rel.endswith(".py"):
            got = py_enclosing(lines, n)
            return (got[0], got[2] - got[1] + 1, None) if got else None
        if rel.endswith(".go"):
            d = q.enclosing_declaration(lines, n)
            if d is None:
                return None
            dup = [x.name for x in q.go_decls(lines)].count(d.name)
            return d.name, d.end - d.start + 1, dup
        return False

    got = decl_at(at)
    if got is False:
        return "hand", "no symbol form exists for this file type (a .sh line citation stays a line number)", None, cited, how, None
    if got is None:
        return "hand", "outside any declaration (package clause, import block, file-level comment, or a gap between declarations)", None, cited, how, None
    name, n_len, dup = got
    if rel.endswith(".go") and (name == "init" or dup != 1):
        return "hand", f"`{name}` is not unique in the file ({dup} declarations), so it names nothing in particular", None, cited, how, None
    if how == "unkeyed" and not _prose_names(name, prose):
        return "hand", f"the index recorded no content for this number, and the prose does not name `{name}`, the declaration it lands in now", None, cited, how, None
    naive = decl_at(line)
    return "ok", name, n_len, cited, how, (naive[0] if naive else None)


def convert_text(text, rel_doc, classify_fn, allow=()):
    """(new_text, events). An event is ("hand", row) for a citation left alone or ("ok", review_row_or_None, drift_row_or_None) for one converted.
    A list or range is one unit: every line in it must resolve to a declaration or the whole citation is left for hand conversion (half a list
    converted would orphan the rest); the names are de-duplicated in order and written `path:A, path:B` when they differ."""
    events = []

    def repl(m):
        rel, a1, a2, extra = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if rel in allow or rel.split("/")[0] in allow:
            return m.group(0)
        old = m.group(0)
        prose = text[max(0, m.start() - 320):m.end() + 320]
        items = [a1] + [int(x) for x, _ in EXTRA_RE.findall(extra)]
        names, longest, cited, drift = [], 0, "", []
        for k, ln in enumerate(items):
            status, name, n_len, c, how, naive = classify_fn(rel, ln, rel_doc, k == 0, prose)
            if status == "hand":
                events.append(("hand", (rel_doc, old, name, c[:100])))
                return old
            if name not in names:
                names.append(name)
            if n_len and n_len > longest:
                longest, cited = n_len, c
            cited = cited or c
            if naive and naive != name:
                drift.append(f"{naive} -> {name} ({how})")
        new = ", ".join(f"{rel}:{n}" for n in names)
        note = []
        if a2 or extra:
            note.append("range" if a2 and not extra else "list")
        if len(names) > 1:
            note.append(f"{len(names)} declarations")
        row = (rel_doc, old, new, str(longest), cited[:100], "+".join(note)) if longest > REVIEW_LINES else None
        drow = (rel_doc, old, new, "; ".join(drift)) if drift else None
        events.append(("ok", row, drow))
        return new

    return CITE_RE.sub(repl, text), events


def read_tsv(path):
    if not path.exists():
        return []
    return [tuple(l.rstrip("\n").split("\t")) for l in path.read_text().split("\n")[1:] if l.strip()]


def write_tsv(path, header, rows):
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(q.ROOT / "docs" / "measurements" / "code-comments-2026-10"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)

    targets = []
    for d in (pathlib.Path(x) for x in a.docs) if a.docs else q.live_docs():
        d = d if d.is_absolute() else q.ROOT / d
        rel = str(d.relative_to(q.ROOT))
        if d == q.QUEUE and not a.docs:
            pass  # QUEUE.md is live and is converted like any other
        if is_record(rel):
            continue
        text = d.read_text(errors="replace")
        if q.pinned_marker(text)[0]:
            continue
        targets.append((d, rel))

    review, hand, drift = [], [], []
    stats = collections.Counter()
    per_doc = collections.Counter()
    allow = q.path_allowlist()
    for d, rel_doc in targets:
        text = d.read_text()
        body_end = len(text)
        if d == q.QUEUE and q.MARK_BEGIN in text:
            body_end = text.index(q.MARK_BEGIN)
        head, tail = text[:body_end], text[body_end:]
        new_head, ev = convert_text(head, rel_doc, classify, allow)
        for e in ev:
            if e[0] == "hand":
                stats["hand"] += 1
                hand.append(e[1])
            else:
                stats["converted"] += 1
                per_doc[rel_doc] += 1
                if e[1]:
                    stats["review"] += 1
                    review.append(e[1])
                if e[2]:
                    stats["drifted"] += 1
                    drift.append(e[2])
        n_anchor = len(ANCHOR_RE.findall(new_head))
        if n_anchor:
            new_head = ANCHOR_RE.sub(lambda m: m.group(1) + ")", new_head)
            stats["anchors"] += n_anchor
        if new_head + tail != text and not a.dry_run:
            d.write_text(new_head + tail)

    # merge the lists with what earlier runs wrote (a converted citation is gone from the docs, so the first run's rows must survive)
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        rv = {(r[0], r[1], r[2]): r for r in read_tsv(out / "review-list.tsv")}
        for r in review:
            rv[(r[0], r[1], r[2])] = r
        write_tsv(out / "review-list.tsv", ("doc", "old_cite", "new_cite", "decl_lines", "cited_line", "note"), sorted(rv.values()))
        dr = {(r[0], r[1], r[2]): r for r in read_tsv(out / "drift-list.tsv")}
        for r in drift:
            dr[(r[0], r[1], r[2])] = r
        write_tsv(out / "drift-list.tsv", ("doc", "old_cite", "new_cite", "stale_number_would_have_named -> resolved_by_recorded_content"), sorted(dr.values()))
        hd = {(r[0], r[1]): r for r in read_tsv(out / "hand-list.tsv")}
        for r in hand:
            hd[(r[0], r[1])] = r
        # a hand-list row whose old citation is no longer in its doc has been converted by hand since: drop it
        live = {}
        for r in list(hd.values()):
            p = q.ROOT / r[0]
            if p.is_file() and r[1] in p.read_text(errors="replace"):
                live[(r[0], r[1])] = r
        write_tsv(out / "hand-list.tsv", ("doc", "old_cite", "reason", "cited_line"), sorted(live.values()))
    print(f"docs examined: {len(targets)}; converted: {stats['converted']} in {len(per_doc)} doc(s); "
          f"left for hand conversion: {stats['hand']}; whose stale number named a DIFFERENT declaration than the recorded content: {stats['drifted']}; in a declaration over {REVIEW_LINES} lines (review list): {stats['review']}; "
          f"#L link anchors stripped: {stats['anchors']}{' (DRY RUN: nothing written)' if a.dry_run else ''}")
    for rel, n in per_doc.most_common(8):
        print(f"  {n:4d}  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
