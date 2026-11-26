#!/usr/bin/env python3
"""LS6 (aikit's WeightMat door): rewrite goinfer's calls of the names steps 1 and 2 replace.

Usage: python3 scripts/aikit_surface/ls6_fix.py [--check] RULES... -- ROOT...
       (today: scripts/aikit_surface/int4_layout.rules scripts/aikit_surface/batch.rules)

  1. The calls that map one to one are `gofmt -r` rules, one a line, applied in file order. Step 1
     (int4_layout.rules): `Int4Row4Usable` becomes a comparison of `PreferredInt4Layout` with `LayoutRow4`, and the
     repacked-only constructors and in-place repacks take their layout. Step 2 (batch.rules): the op types become
     `BatchOp`, a W8A8 op literal built from one weight becomes `BatchOp{W: w, Dst: d}`, and both batch calls become
     `MatmulBTBatchInto`.
  2. `WrapInt4F16` has no single-value replacement (`WrapInt4` also reports ok, always true for `LayoutCanonical`). The
     two statement forms it appears in are rewritten here: `x = linalg.WrapInt4F16(...)` (or `:=`) becomes
     `x, _ = linalg.WrapInt4(..., linalg.LayoutCanonical)`, and `return linalg.WrapInt4F16(...)` (with or without further
     results) gets a local first.
  3. Every call of a replaced name left after that is printed as a site to edit by hand (README.md lists them).

Comments and strings are not rewritten. A second run changes nothing. --check reports, writes nothing, and exits 1 if
anything would change.
"""

import os
import re
import subprocess
import sys

SKIP_DIRS = {"testdata", "vendor", ".git", "node_modules"}
REPLACED = [
    "WrapInt4F16", "WrapInt4Row4", "WrapInt4Row4F16", "WrapInt4Row4Only", "WrapInt4Row4OnlyF16",
    "WrapInt4SplitHalfOnly", "WrapInt4SplitHalfOnlyF16", "RepackInt4Row4InPlace", "RepackInt4Row4InPlaceF16",
    "RepackInt4SplitHalfInPlace", "RepackInt4SplitHalfInPlaceF16", "Int4Row4Usable", "Int4SplitHalfUsable",
    "W8A8Op", "W4A8Op", "MatmulBTW8A8Batch", "MatmulBTW4A8Batch",
]
# The old f32-scale WrapInt4(q4, q4s, rows, cols, group) had five arguments; the new one has six. A five-argument call
# left anywhere is a site by hand.
OLD_WRAPINT4 = re.compile(r"\blinalg\.WrapInt4\((?:[^()]|\([^()]*\))*\)")


def rules(paths):
    out = []
    for ln in (ln for p in paths for ln in open(p)):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def go_files(roots):
    for root in roots:
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith((".", "_"))]
            for f in files:
                if f.endswith(".go"):
                    yield os.path.join(d, f)


def code_only(src):
    """src with comments and string/rune literals blanked, for identifier searches."""
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|`[^`]*`|\'(?:\\.|[^\'\\\n])*\'', " ", src, flags=re.S)


ASSIGN = re.compile(r"^(\s*)([\w.*\[\]]+) (:?=) linalg\.WrapInt4F16\((.*)\)$")
RETURN = re.compile(r"^(\s*)return linalg\.WrapInt4F16\((.*)\)((?:, \w+)*)$")


def statement_forms(src):
    out = []
    for ln in src.split("\n"):
        m = ASSIGN.match(ln)
        if m:
            ind, lhs, op, args = m.groups()
            out.append(f"{ind}{lhs}, _ {op} linalg.WrapInt4({args}, linalg.LayoutCanonical)")
            continue
        m = RETURN.match(ln)
        if m:
            ind, args, rest = m.groups()  # rest: a further result, as in return linalg.WrapInt4F16(...), nil
            out.append(f"{ind}wm4, _ := linalg.WrapInt4({args}, linalg.LayoutCanonical)")
            out.append(f"{ind}return wm4{rest}")
            continue
        out.append(ln)
    return "\n".join(out)


def gofmt_rule(path, rule, src):
    p = subprocess.run(["gofmt", "-r", rule], input=src, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{path}: gofmt -r {rule!r}: {p.stderr.strip()}")
    return p.stdout


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    check = "--check" in sys.argv[1:]
    if "--" not in args:
        sys.exit("usage: ls6_fix.py [--check] RULES... -- ROOT...")
    i = args.index("--")
    rs, roots = rules(args[:i]), args[i + 1:] or ["."]
    changed, by_hand = [], []
    for path in go_files(roots):
        src = open(path).read()
        if not re.search(r"\blinalg\.(?:" + "|".join(REPLACED) + r")\b", code_only(src)):
            continue
        new = src
        for r in rs:
            name = re.match(r"!?linalg\.(\w+)", r).group(1)
            if re.search(r"\blinalg\." + name + r"\b", code_only(new)):
                new = gofmt_rule(path, r, new)
        new = statement_forms(new)
        if new != src:
            changed.append(path)
            if not check:
                open(path, "w").write(new)
        code = code_only(new)
        for i, ln in enumerate(code.split("\n"), 1):
            for name in REPLACED:
                if re.search(r"\blinalg\." + name + r"\b", ln):
                    by_hand.append(f"{path}:{i}: linalg.{name}")
            if OLD_WRAPINT4.search(ln) and ln.count(",") < 5:
                by_hand.append(f"{path}:{i}: linalg.WrapInt4 with f32 scales")
    for p in changed:
        print(("would change " if check else "changed ") + p)
    for s in by_hand:
        print("by hand: " + s)
    if check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
