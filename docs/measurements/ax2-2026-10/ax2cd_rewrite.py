#!/usr/bin/env python3
"""AX2c and AX2d: switch one backend module from aikit's gpu device module to its half of goinfer's, or check a commit.

    ax2cd_rewrite.py <metal|cuda> rewrite            # in the repository root: rewrites that module's .go files in place
    ax2cd_rewrite.py <metal|cuda> check <commit>     # <commit> is exactly this rewrite applied to its parent

The import becomes `gpu "github.com/townsendmerino/goinfer/<module>/internal/gpudev"`: the package is named gpudev, and
the alias keeps every body unchanged. gofmt runs on each rewritten file.
"""
import re
import subprocess
import sys

OLD = re.compile(r'^(\s*)(\w+ )?"github\.com/townsendmerino/aikit/gpu"', re.M)


def git(*args, binary=False):
    out = subprocess.run(["git", *args], capture_output=True, check=True).stdout
    return out if binary else out.decode()


def rewrite(module, src: bytes) -> bytes:
    new = f'"github.com/townsendmerino/goinfer/{module}/internal/gpudev"'
    text = OLD.sub(lambda m: f'{m.group(1)}{m.group(2) or "gpu "}{new}', src.decode())
    return subprocess.run(["gofmt"], input=text.encode(), capture_output=True, check=True).stdout


def importers(module, rev=None):
    args = ["grep", "-l", "-E", r'"github\.com/townsendmerino/aikit/gpu"'] + ([rev] if rev else []) + ["--", f"{module}/*.go"]
    names = subprocess.run(["git", *args], capture_output=True).stdout.decode().split()
    return [n.split(":", 1)[1] if rev else n for n in names]


def main():
    module, mode = sys.argv[1], sys.argv[2]
    if mode == "rewrite":
        files = importers(module)
        for f in files:
            with open(f, "rb") as fh:
                data = rewrite(module, fh.read())
            with open(f, "wb") as fh:
                fh.write(data)
        print(f"rewrote {len(files)} files in {module}")
        return
    commit = sys.argv[3]
    parent = commit + "^"
    want = set(importers(module, parent))
    changed = set(git("diff", "--name-only", parent, commit).split())
    bad = 0
    for f in sorted(want | changed):
        if f not in want:
            print(f"EXTRA: {f}")
            bad += 1
        elif git("show", f"{commit}:{f}", binary=True) != rewrite(module, git("show", f"{parent}:{f}", binary=True)):
            print(f"DIFFERS: {f}")
            bad += 1
    for f in importers(module, commit):
        print(f"STILL IMPORTS aikit/gpu: {f}")
        bad += 1
    print(f"{module}: {len(want)} files rewritten at {commit[:12]}, {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
