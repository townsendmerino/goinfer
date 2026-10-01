#!/usr/bin/env python3
"""Writes previous/next front matter onto a staged copy of the book (scripts/build_book_src.sh).

Reading order is the front page (index.md), then the numbered pages in file order (the eleven chapters, then the
glossary). Each page gets bk_prev_url/bk_prev_title and bk_next_url/bk_next_title, and every page but the front page
gets bk_chapter, which the book's layout (docs/book-search/default.html) uses to show the masthead.

    scripts/book_pager.py STAGED_DIR
"""
import json
import os
import re
import sys


def pages(src):
    names = sorted(n for n in os.listdir(src) if re.match(r"^\d{2}-.+\.md$", n))
    out = [("index.md", "Contents")]
    for n in names:
        with open(os.path.join(src, n), encoding="utf-8") as f:
            first = f.readline().strip()
        if not first.startswith("# "):
            sys.exit(f"{n}: the first line is not a '# Title' heading, so the pager has no title for it")
        out.append((n, first[2:].strip()))
    return out


def url(name):
    return "./" if name == "index.md" else name[:-3] + ".html"


def main(src):
    ps = pages(src)
    for i, (name, _) in enumerate(ps):
        path = os.path.join(src, name)
        with open(path, encoding="utf-8") as f:
            body = f.read()
        if body.startswith("---\n"):
            sys.exit(f"{name} already has front matter; merge the book's prev/next keys into it instead of overwriting")
        fm = ["---"]
        if name != "index.md":
            fm.append("bk_chapter: true")
        if i > 0:
            fm += ["bk_prev_url: " + json.dumps(url(ps[i - 1][0])), "bk_prev_title: " + json.dumps(ps[i - 1][1], ensure_ascii=False)]
        if i + 1 < len(ps):
            fm += ["bk_next_url: " + json.dumps(url(ps[i + 1][0])), "bk_next_title: " + json.dumps(ps[i + 1][1], ensure_ascii=False)]
        fm.append("---")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(fm) + "\n" + body)
    print(f"book pager: previous/next written on {len(ps)} pages")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
