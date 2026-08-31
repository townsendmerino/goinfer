#!/usr/bin/env python3
"""book_link_lint.py — every link out of docs/book/ must still point at something.

WHY THIS EXISTS. The book carries ~60 links into the repo's own source, written as
absolute GitHub blob URLs because the book is ALSO published to GitHub Pages, where
the Jekyll build stages docs/book/*.md into a flat directory and a relative path like
../../decoder/model.go resolves to nothing. Absolute URLs are the only form that works
in both places — and absolute URLs into a moving repo are exactly the kind of citation
that rots silently: `git mv decoder/model.go decoder/forward.go` leaves every link
syntactically perfect and every one of them a 404.

Deliberately NOT line-anchored. `queue_citation_lint.py` maintains path:line citations
and re-keys them by content, which is the right tool where a specific LINE carries the
argument. The book cites whole FILES ("the loop lives here"), so line numbers would add
a re-keying chore on every unrelated edit and buy the reader nothing. File-level links
only rot on rename, which is rare — and which this catches.

WHY IT RUNS IN ci.yml AND NOT IN book-pages.yml. book-pages triggers on
`docs/book/**` only. The failure this guards against originates in `decoder/`, so
the publish workflow would never fire for it: the book would keep deploying green,
with dead links, until a reader clicked one. It has to run where every push runs.

Checks, in order:
  1. Every GitHub blob link into THIS repo resolves to a tracked path.
  2. Every relative link/image (the .svg figures) resolves inside docs/book/.
  3. The scan found a non-trivial number of links at all.

(3) is not padding. An empty result from a link checker is indistinguishable from a
checker whose glob stopped matching, and this repo has had exactly that class of
silent-green failure before. If the count ever drops to zero the lint fails and says
so, rather than printing a reassuring nothing.

Exit 0 = sound. Exit 1 = at least one dead link (or a vacuous scan).
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BOOK = ROOT / "docs" / "book"

# Links into this repo's source, as written by the book.
SELF_BLOB = re.compile(
    r"\]\(https://github\.com/townsendmerino/goinfer/blob/main/([^)#\s]+)(?:#[^)\s]*)?\)"
)
# Relative links and images: ![alt](./fig.svg) and [text](./other.md).
RELATIVE = re.compile(r"\]\((\./[^)\s]+)\)")

# A scan finding fewer than this many links means the glob or the pattern broke,
# not that the book stopped citing code. Set well below the real count (~60) so
# ordinary editing never trips it.
MIN_LINKS = 20


def main() -> int:
    if not BOOK.is_dir():
        sys.stderr.write(f"book_link_lint: no book at {BOOK}\n")
        return 1

    dead: list[str] = []
    total = 0

    for md in sorted(BOOK.glob("*.md")):
        body = md.read_text()

        for m in SELF_BLOB.finditer(body):
            total += 1
            target = ROOT / m.group(1)
            if not target.exists():
                dead.append(
                    f"  {md.name}: blob link -> {m.group(1)} does NOT exist "
                    f"(renamed or deleted; fix the link, do not delete the citation)"
                )

        for m in RELATIVE.finditer(body):
            total += 1
            rel = m.group(1)
            if not (BOOK / rel).exists():
                dead.append(
                    f"  {md.name}: relative link -> {rel} does NOT exist "
                    f"(figures must live in docs/book/ so the Pages build stages them flat)"
                )

    if dead:
        sys.stderr.write("book_link_lint: dead links out of docs/book/:\n")
        sys.stderr.write("\n".join(dead) + "\n")
        return 1

    if total < MIN_LINKS:
        sys.stderr.write(
            f"book_link_lint: only {total} links found, expected >= {MIN_LINKS}.\n"
            f"That is far more likely to mean this lint stopped matching than that the\n"
            f"book stopped citing code. Check the globs and the regexes before lowering\n"
            f"MIN_LINKS — a link checker that silently checks nothing is worse than none.\n"
        )
        return 1

    print(f"book_link_lint: {total} links out of docs/book/, all resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
