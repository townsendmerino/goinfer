#!/usr/bin/env python3
"""readme_counts_check.py — do docs/README.md's file counts match what's actually on disk?

docs/README.md's own skill notes (this repo's doc-review skill) already name this as "Known drift
to expect": these counts "drift constantly; recount from source." They were wrong when this script
was first run, 2026-09-13, not hypothetically: `measurements/` claimed **151**, actually **75**
(nearly half stale — likely an old count from before a consolidation, never revisited);
`completed/` claimed **96**, actually **117** (three archival sweeps landed the same day the
number was last touched); `prompts/` claimed **26**, actually **4** (a same-day sweep archived 21
of 25 prompt briefs and nobody went back to fix the one line that said how many were left).

Each count below is a small, separate claim in different prose shapes (a section heading's
parenthetical, an inline mention mid-sentence, a bullet), so this is a table of (regex, what it
should equal), not one generic rule. New sections in docs/README.md that carry a count need a new
entry added here by hand — there is no way to discover "this is a count claim" mechanically without
false-positiving on every other parenthetical number in the file (a date, a version, a file size).

Deliberately NOT checked: "cited from 88 code comments" (docs/README.md's Design records section) —
which task docs' comments count, and by what exact grep pattern, is a real judgment call, not a
single directory listing; automating it would need a precise, arguable definition this script
should not invent unilaterally. The opening line's "~270 files" is deliberately approximate (the
tilde says so) and is reported as a soft note, not a hard mismatch.

This is a read-only report, not a push gate and not an auto-rewriter — prose around a stale number
needs a human to phrase the correction, the same way every other doc-review edit in this repo does.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "docs" / "README.md"


def count_md(relglob: str) -> int:
    return len(list((ROOT / "docs").glob(relglob.removeprefix("docs/"))))


# (label, regex with the count(s) as named groups, {group_name: actual_count_fn})
CHECKS = [
    (
        "Design records heading — total",
        re.compile(r"## Design records .*?\((\d+):"),
        lambda: count_md("docs/tasks/*.md") + count_md("docs/tasks/parked/*.md"),
    ),
    (
        "Design records heading — tasks/",
        re.compile(r"## Design records .*?\(\d+: (\d+) in `tasks/`"),
        lambda: count_md("docs/tasks/*.md"),
    ),
    (
        "Design records heading — tasks/parked/",
        re.compile(r"## Design records .*?in `tasks/`, (\d+) in `tasks/parked/`"),
        lambda: count_md("docs/tasks/parked/*.md"),
    ),
    (
        "`spec/` inline count",
        re.compile(r"`spec/` \((\d+)\)"),
        lambda: count_md("docs/spec/*.md"),
    ),
    (
        "Evidence heading — measurements/",
        re.compile(r"## Evidence .*?`measurements/` \((\d+)\)"),
        lambda: count_md("docs/measurements/*.md"),
    ),
    (
        "Archive heading — completed/",
        re.compile(r"## Archive .*?`completed/` \((\d+)\)"),
        lambda: count_md("docs/completed/*.md"),
    ),
    (
        "`prompts/` bullet",
        re.compile(r"`prompts/` \((\d+)\)"),
        lambda: count_md("docs/prompts/*.md"),
    ),
    (
        "`releases/` bullet",
        re.compile(r"`releases/` \((\d+)\)"),
        lambda: count_md("docs/releases/*.md"),
    ),
]


def main() -> int:
    if not README.exists():
        print(f"readme_counts_check: {README} not found.")
        return 1
    text = README.read_text(encoding="utf-8", errors="replace")

    stale = []
    for label, pattern, actual_fn in CHECKS:
        m = pattern.search(text)
        if not m:
            print(f"SKIP  {label}: pattern not found in docs/README.md (did the wording change?)")
            continue
        stated = int(m.group(1))
        actual = actual_fn()
        if stated != actual:
            stale.append((label, stated, actual))

    m = re.search(r"holds ~(\d+) files", text)
    total_actual = len(list((ROOT / "docs").rglob("*.md")))
    if m:
        stated = int(m.group(1))
        if abs(stated - total_actual) / max(total_actual, 1) > 0.10:
            print(
                f"NOTE  opening line says ~{stated} files, docs/ actually has {total_actual} "
                "(off by more than 10% — still just an approximation, worth a glance)"
            )

    if not stale:
        print("readme_counts_check: every checked count in docs/README.md matches disk.")
        return 0

    print("STALE counts in docs/README.md:")
    for label, stated, actual in stale:
        print(f"  {label}: says {stated}, actually {actual}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
