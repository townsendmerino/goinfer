#!/usr/bin/env python3
"""The book and the site must wear the same tokens (docs/tasks/task-site-2026-09.md, 4a). The book's CSS lives in
docs/book-search/head-custom.html (the include book-pages.yml ships); the site's reference is site/mockups/*.html. This
fails if a colour or font token differs, in the light block or the dark one.

    python3 -B scripts/test_book_tokens.py

Once the site generator exists and owns a tokens file, point SITE at it instead of the mockups.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "docs", "book-search", "head-custom.html")
SITE = [os.path.join(ROOT, "site", "mockups", n) for n in ("home.html", "models.html", "different-confidence.html")]
NAMES = ["ground", "surface", "sunk", "line", "ink", "ink-2", "muted", "accent", "accent-soft", "on-accent",
         "f-display", "f-body", "f-mono"]


def block(text, start):
    """The text between the braces that open at or after `start`, matched."""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        depth += text[j] == "{"
        depth -= text[j] == "}"
        if depth == 0:
            return text[i + 1:j]
    raise ValueError("unbalanced braces")


def decls(body):
    out = {}
    for m in re.finditer(r"--([a-z0-9-]+)\s*:\s*([^;}]+)", body):
        v = re.sub(r"\s+", " ", m.group(2).strip().strip(";")).lower()
        out[m.group(1)] = re.sub(r"\s*,\s*", ",", v)       # "a, b" and "a,b" are the same font list
    return out


def tokens(path):
    t = open(path, encoding="utf-8").read()
    root = t.index(":root")
    light = decls(block(t, root))
    dm = t.index("prefers-color-scheme: dark", root)
    inner = block(t, dm)                     # the @media body, which holds the :root rule
    dark = decls(block(inner, inner.index(":root")))
    return light, dark


class BookTokens(unittest.TestCase):
    def test_book_matches_the_site(self):
        bl, bd = tokens(BOOK)
        for path in SITE:
            sl, sd = tokens(path)
            name = os.path.basename(path)
            for n in NAMES:
                self.assertEqual(bl.get(n), sl.get(n), f"light --{n}: book {bl.get(n)!r} vs {name} {sl.get(n)!r}")
            for n in NAMES[:10]:
                self.assertEqual(bd.get(n), sd.get(n), f"dark --{n}: book {bd.get(n)!r} vs {name} {sd.get(n)!r}")

    def test_the_workflows_guards_still_hold(self):
        # book-pages.yml refuses head-custom.html with a Liquid tag or with more than one HTML comment.
        t = open(BOOK, encoding="utf-8").read()
        self.assertNotIn("{%", t)
        self.assertEqual((t.count("<!--"), t.count("-->")), (1, 1))

    def test_every_token_is_present(self):
        bl, bd = tokens(BOOK)
        self.assertTrue(all(n in bl for n in NAMES), [n for n in NAMES if n not in bl])
        self.assertTrue(all(n in bd for n in NAMES[:10]), [n for n in NAMES[:10] if n not in bd])


if __name__ == "__main__":
    unittest.main()
