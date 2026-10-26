#!/usr/bin/env python3
"""AX2b: switch every importer of aikit's vision, audio and gpu/visioncuda packages to goinfer's copies, or check that a
commit did.

    ax2_rewrite.py rewrite            # in the repository root: rewrites the tracked .go files in place, then gofmt
    ax2_rewrite.py check <commit>     # gate AX2b-N: <commit> is exactly this rewrite applied to its parent

The rewrite is the import path and nothing else; gofmt re-sorts an import block whose order the new path changes.
"""
import subprocess
import sys

# aikit's gpu/visioncuda registers into aikit's vision, which nothing reads after the switch; its copy registers into
# goinfer's.
OLD_NEW = [
    ('"github.com/townsendmerino/aikit/gpu/visioncuda"', '"github.com/townsendmerino/goinfer/cuda/internal/visioncuda"'),
    ('"github.com/townsendmerino/aikit/vision"', '"github.com/townsendmerino/goinfer/internal/vision"'),
    ('"github.com/townsendmerino/aikit/audio"', '"github.com/townsendmerino/goinfer/internal/audio"'),
]


def git(*args, binary=False):
    out = subprocess.run(["git", *args], capture_output=True, check=True).stdout
    return out if binary else out.decode()


def rewrite(src: bytes) -> bytes:
    text = src.decode()
    for old, new in OLD_NEW:
        text = text.replace(old, new)
    return subprocess.run(["gofmt"], input=text.encode(), capture_output=True, check=True).stdout


def importers(rev=None):
    pattern = "|".join(old.strip('"').replace(".", r"\.") for old, _ in OLD_NEW)
    args = ["grep", "-l", "-E", f'"({pattern})"'] + ([rev] if rev else []) + ["--", "*.go"]
    r = subprocess.run(["git", *args], capture_output=True)
    names = r.stdout.decode().split()
    return [n.split(":", 1)[1] if rev else n for n in names]


def main():
    if sys.argv[1] == "rewrite":
        files = importers()
        for f in files:
            with open(f, "rb") as fh:
                data = rewrite(fh.read())
            with open(f, "wb") as fh:
                fh.write(data)
        print(f"rewrote {len(files)} files")
        return
    commit = sys.argv[2]
    parent = commit + "^"
    want = set(importers(parent))
    changed = set(git("diff", "--name-only", parent, commit).split())
    bad = 0
    for f in sorted(want | changed):
        if f not in want:
            print(f"EXTRA: {f} changed but imported neither package at the parent")
            bad += 1
            continue
        if git("show", f"{commit}:{f}", binary=True) != rewrite(git("show", f"{parent}:{f}", binary=True)):
            print(f"DIFFERS: {f}")
            bad += 1
    left = importers(commit)
    for f in left:
        print(f"STILL IMPORTS aikit: {f}")
        bad += 1
    print(f"{len(want)} files rewritten at {commit[:12]}, {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
