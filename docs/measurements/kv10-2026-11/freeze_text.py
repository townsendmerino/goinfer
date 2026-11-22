#!/usr/bin/env python3
"""Writes KV10's frozen prompt text: the text backend/cuda/kv3_longprefill_test.go:kv10Text built from the tree at a commit.

kv10Text at 157e2ca29 (the registration's first reading; its text sources are identical at 5ecc0d73d) concatenated the tracked,
non-test files matching decoder/*.go, docs/*.md and backend/cuda/*.go, each glob in sorted order, stopping once the text passed
3,000,000 bytes, and took the first N tokens. This script rebuilds those bytes from git, keeps the whole files up to the first one
that ends past --min-bytes (enough for the 131,000-token depth plus slack), and writes them gzipped with no name and mtime 0, so the
output is byte-reproducible. Owner decision: Francis, 2026-11-22, option A (docs/measurements/kv10-2026-11/README.md, "The frozen
text").

    python3 docs/measurements/kv10-2026-11/freeze_text.py 157e2ca29 backend/cuda/testdata/kv10_text_157e2ca29.txt.gz
"""
import argparse
import gzip
import hashlib
import io
import subprocess


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def files(rev):
    out = []
    for d, ext in (("decoder/", ".go"), ("docs/", ".md"), ("backend/cuda/", ".go")):
        names = git("ls-tree", "--name-only", rev, d).decode().split("\n")
        names = sorted(n for n in names if n.endswith(ext) and "/" not in n[len(d):])
        out += [n for n in names if not n.endswith("_test.go")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rev")
    ap.add_argument("out")
    ap.add_argument("--min-bytes", type=int, default=500_000)
    a = ap.parse_args()
    text, used = bytearray(), []
    for f in files(a.rev):
        text += git("show", f"{a.rev}:{f}")
        used.append(f)
        if len(text) > a.min_bytes:
            break
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as g:
        g.write(bytes(text))
    with open(a.out, "wb") as f:
        f.write(buf.getvalue())
    print(f"{a.rev}: {len(used)} files ({used[0]} .. {used[-1]}), {len(text)} bytes, sha256 {hashlib.sha256(text).hexdigest()}, "
          f"{len(buf.getvalue())} bytes gzipped -> {a.out}")


if __name__ == "__main__":
    main()
