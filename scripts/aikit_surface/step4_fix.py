#!/usr/bin/env python3
"""LS6 step 4 (aikit's raw-slice int8 matmuls unexported): route goinfer's calls through a WeightMat.

Usage: python3 scripts/aikit_surface/step4_fix.py [--check] ROOT...

  1. A call whose weight is already a WeightMat w (decoder's matmul helpers: the raw arrays read back from w, then
     MatmulBTW8A8Into/MatmulBTQ8Into(ws, a, q, s, dst, M, w.Cols(), w.Rows())) becomes w.MatmulBTInto(ws, a, dst, M):
     the same kernel, since MatmulBTInto dispatches a W8A8 weight to MatmulBTW8A8Into and a weight-only one to
     MatmulBTQ8Into. The raw arrays it read for the call may then be unused: README.md lists those edits.
  2. Every other call of MatmulBTW8A8, MatmulBTW8A8Into, MatmulBTQ8 or MatmulBTQ8Into becomes a call of a helper of
     the same package (aikitMatmulW8A8, ...) that wraps the int8 rows and scales in a WeightMat (WrapInt8 copies
     nothing) and runs the same kernel. The script writes the package's aikit_matmul_helpers.go (or _test.go, when
     only tests call them) with the helpers that package uses.

The W4A8 forms take f32 scales, which a WeightMat stores as binary16, so their few callers are edited by hand.
A second run changes nothing. --check reports, writes nothing, and exits 1 if anything would change.
"""

import os
import re
import subprocess
import sys

SKIP_DIRS = {"testdata", "vendor", ".git", "node_modules"}

WEIGHTMAT_RULES = [
    "linalg.MatmulBTW8A8Into(s, a, q, c, d, m, w.Cols(), w.Rows()) -> w.MatmulBTInto(s, a, d, m)",
    "linalg.MatmulBTQ8Into(s, a, q, c, d, m, w.Cols(), w.Rows()) -> w.MatmulBTInto(s, a, d, m)",
]

# name -> (helper, w8a8, takes a Workspace)
RAW = {
    "MatmulBTW8A8": ("aikitMatmulW8A8", True, False),
    "MatmulBTW8A8Into": ("aikitMatmulW8A8Into", True, True),
    "MatmulBTQ8": ("aikitMatmulQ8", False, False),
    "MatmulBTQ8Into": ("aikitMatmulQ8Into", False, True),
}


def helper_src(name):
    h, w8a8, ws = RAW[name]
    kind = "W8A8 (int8 activations)" if w8a8 else "weight-only Q8 (f32 activations)"
    if ws:
        return f'''// {h} is dst[M, N] = a[M, K] · Wᵀ for int8 weights q [N, K] with per-row scales, {kind}, through ws: what
// linalg.{name} computed, through a WeightMat over the same slices (aikit LS6 unexported the raw-slice form).
func {h}(ws *linalg.Workspace, a []float32, q []int8, scales, dst []float32, M, K, N int) {{
	w := linalg.WrapInt8(q, scales, N, K, {str(w8a8).lower()})
	w.MatmulBTInto(ws, a, dst, M)
}}
'''
    return f'''// {h} is dst[M, N] = a[M, K] · Wᵀ for int8 weights q [N, K] with per-row scales, {kind}: what linalg.{name}
// computed, through a WeightMat over the same slices (aikit LS6 unexported the raw-slice form).
func {h}(a []float32, q []int8, scales, dst []float32, M, K, N int) {{
	w := linalg.WrapInt8(q, scales, N, K, {str(w8a8).lower()})
	w.MatmulBT(a, dst, M)
}}
'''


def go_files(roots):
    for root in roots:
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith((".", "_"))]
            for f in files:
                if f.endswith(".go"):
                    yield os.path.join(d, f)


def code_only(src):
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|`[^`]*`|\'(?:\\.|[^\'\\\n])*\'', " ", src, flags=re.S)


def gofmt_rule(path, rule, src):
    p = subprocess.run(["gofmt", "-r", rule], input=src, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{path}: gofmt -r {rule!r}: {p.stderr.strip()}")
    return p.stdout


def package_of(src):
    return re.search(r"^package (\w+)", src, re.M).group(1)


def main():
    args = sys.argv[1:]
    check = "--check" in args
    roots = [a for a in args if a != "--check"] or ["."]
    changed = []
    # dir -> {"prod": set(names), "test": set(names), "pkg": name}
    needs = {}
    for path in go_files(roots):
        if os.path.basename(path).startswith("aikit_matmul_helpers"):
            continue
        src = open(path).read()
        code = code_only(src)
        if not re.search(r"\blinalg\.(?:" + "|".join(RAW) + r")\(", code) and not re.search(r"\baikitMatmul(?:W8A8|Q8)(?:Into)?\(", code):
            continue
        new = src
        for r in WEIGHTMAT_RULES:
            name = re.match(r"linalg\.(\w+)", r).group(1)
            if re.search(r"\blinalg\." + name + r"\b", code_only(new)):
                new = gofmt_rule(path, r, new)
        for name, (h, _, _) in RAW.items():
            if re.search(r"\blinalg\." + name + r"\(", code_only(new)):
                new = gofmt_rule(path, f"linalg.{name} -> {h}", new)
        if new != src and not re.search(r"\blinalg\.", code_only(new)):
            # The file's last linalg selector went to a helper: drop the import (gofmt joins the group back).
            new = re.sub(r'^\s*"github.com/townsendmerino/aikit/linalg"\n', "", new, count=1, flags=re.M)
            new = re.sub(r'^import "github.com/townsendmerino/aikit/linalg"\n', "", new, count=1, flags=re.M)
            new = subprocess.run(["gofmt"], input=new, capture_output=True, text=True, check=True).stdout
        used = {n for n, (h, _, _) in RAW.items() if re.search(r"\b" + h + r"\(", code_only(new))}
        if used:
            d = os.path.dirname(path)
            e = needs.setdefault(d, {"prod": set(), "test": set(), "pkg": None})
            pkg = package_of(new)
            if pkg.endswith("_test"):
                sys.exit(f"{path}: an external test package; give it its own helpers by hand")
            e["pkg"] = pkg
            e["test" if path.endswith("_test.go") else "prod"] |= used
        if new != src:
            changed.append(path)
            if not check:
                open(path, "w").write(new)
    for d, e in sorted(needs.items()):
        names = sorted(e["prod"] | e["test"])
        fname = "aikit_matmul_helpers.go" if e["prod"] else "aikit_matmul_helpers_test.go"
        body = f"package {e['pkg']}\n\n// Generated by scripts/aikit_surface/step4_fix.py; regenerated on each run.\n\n"
        body += 'import "github.com/townsendmerino/aikit/linalg"\n\n' + "\n".join(helper_src(n) for n in names)
        out = os.path.join(d, fname)
        other = os.path.join(d, "aikit_matmul_helpers_test.go" if e["prod"] else "aikit_matmul_helpers.go")
        old = open(out).read() if os.path.exists(out) else None
        if old != body or os.path.exists(other):
            changed.append(out)
            if not check:
                open(out, "w").write(body)
                if os.path.exists(other):
                    os.remove(other)
    for p in changed:
        print(("would change " if check else "changed ") + p)
    if check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
