#!/usr/bin/env python3
"""build_search_index.py — emit a client-side search index for the book site.

WHY A GENERATED JSON AND NOT A JEKYLL PLUGIN. GitHub Pages runs Jekyll in safe mode with a
short allowlist of plugins; no search plugin is on it, and the book build (see
.github/workflows/book-pages.yml) enables only jekyll-relative-links. So the index is built
as an ordinary workflow step and the UI is client-side: fetch one JSON, filter in the browser,
no server and no plugin.

GRANULARITY IS THE SECTION, NOT THE PAGE. A chapter is thousands of words; a hit that says only
"chapter 8" makes the reader scan it again. Each record is one heading and the prose under it,
so a result can link to #the-flag and land on the passage.

Usage:  build_search_index.py <staged-dir> <out.json>
"""
import json
import pathlib
import re
import sys

# Fenced code and inline code are dropped: the book quotes Go identifiers and kernel names that
# would otherwise dominate a text search over prose.
FENCE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`]*`")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")   # keep the text, drop the target
MDCHARS = re.compile(r"[*_>#|]+")
WS = re.compile(r"\s+")


def slug(heading: str) -> str:
    """GitHub's heading-anchor rule: lowercase, drop punctuation, spaces to hyphens.

    Matches what jekyll-relative-links + Primer produce, which is what an in-page #anchor has to
    agree with. Getting this wrong yields links that resolve to the page and ignore the fragment
    — working badly rather than failing, so it is worth being exact.
    """
    s = heading.strip().lower()
    s = INLINE.sub(lambda m: m.group(0).strip("`"), s)
    s = LINK.sub(r"\1", s)
    s = re.sub(r"[^\w\s-]", "", s)
    return WS.sub("-", s).strip("-")


def clean(text: str) -> str:
    text = FENCE.sub(" ", text)
    text = INLINE.sub(" ", text)
    text = LINK.sub(r"\1", text)
    text = MDCHARS.sub(" ", text)
    return WS.sub(" ", text).strip()


def main() -> int:
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    records = []
    for md in sorted(src.glob("*.md")):
        # index.md is the chapter table; its own headings are navigation, not content.
        url = "index.html" if md.stem == "index" else f"{md.stem}.html"
        lines = md.read_text(encoding="utf-8").split("\n")
        title = next((l[2:].strip() for l in lines if l.startswith("# ")), md.stem)
        heading, buf = None, []

        def flush():
            body = clean("\n".join(buf))
            if not body and heading is None:
                return
            records.append({
                "chapter": title,
                "url": url + (f"#{slug(heading)}" if heading else ""),
                "heading": heading or title,
                "text": body[:1200],   # enough to match on; the page is the real content
            })

        for line in lines:
            if line.startswith("## "):
                flush()
                heading, buf = line[3:].strip(), []
            elif not line.startswith("# "):
                buf.append(line)
        flush()

    out.write_text(json.dumps(records, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"search index: {len(records)} sections from {len(list(src.glob('*.md')))} pages -> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
