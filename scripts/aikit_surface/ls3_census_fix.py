#!/usr/bin/env python3
"""LS3's follow-up (aikit's CPU flags moved to internal/cpufeat, and LS4's kernel packages): re-key goinfer's two
censuses of aikit's dispatch flags.

Usage: python3 scripts/aikit_surface/ls3_census_fix.py [--check] AIKIT_DIR
       (AIKIT_DIR: the aikit checkout the bump goes to; the script reads where each file now lives)

  1. docs/hardware-coverage.json: the cpu: predicates take cpufeat's exported names (cpu:hasAVX2 -> cpu:HasAVX2, ...).
     embed keeps its own hasAVX2, so the AVX2 entry lists both cpu:HasAVX2 and cpu:hasAVX2.
  2. internal/diag/kernelprint/census_test.go (K3): every exclusion key file|function|condition|verdict is re-keyed.
     A file LS4 moved (linalg/X.go now under linalg/attn, linalg/elem or linalg/bitvec) takes its new path, and the
     tier switch linalg/kernels_<arch>.go|disableTopTier becomes internal/cpufeat/kernels_<arch>.go|DisableTopTier. In
     each condition a flag becomes cpufeat.HasX (HasX inside internal/cpufeat itself). The uncalled map's keys take
     the package their function now lives in.

The code changes that go with this (which packages K3 scans, its predicate map, a cpufeat.HasX selector in its
evaluator, hwcensus reading the go.work aikit and matching Has<X>) are hand edits; README.md lists them. A second run
changes nothing. --check reports, writes nothing, and exits 1 if anything would change.
"""

import json
import os
import re
import sys

FLAGS = ["DotProd", "AVX2", "F16C", "AVX512VNNIVL", "AVX512VNNI", "POPCNT", "Q4KAVX2"]
FLAG_RE = re.compile(r"\bhas(" + "|".join(FLAGS) + r")\b")
KERNEL_PKGS = ["linalg/attn", "linalg/elem", "linalg/bitvec"]
CENSUS = "docs/hardware-coverage.json"
K3 = "internal/diag/kernelprint/census_test.go"


def fix_json(src):
    d = json.loads(src)
    for e in d["entries"]:
        preds = []
        for p in e["predicates"]:
            m = re.fullmatch(r"cpu:has(\w+)", p)
            if m and m.group(1) in FLAGS:
                if m.group(1) == "AVX2":
                    preds += ["cpu:HasAVX2", "cpu:hasAVX2"]  # cpufeat's, and embed's own
                else:
                    preds.append("cpu:Has" + m.group(1))
            elif p not in preds:
                preds.append(p)
        e["predicates"] = list(dict.fromkeys(preds))
    indent = 2 if '\n  "' in src else 1
    return json.dumps(d, indent=indent, ensure_ascii=False) + "\n"


def new_path(aikit, rel):
    """Where aikit file rel (linalg/X.go, embed/X.go) lives now."""
    m = re.fullmatch(r"linalg/kernels_(\w+)\.go", rel)
    if m:
        return "internal/cpufeat/kernels_" + m.group(1) + ".go"
    if os.path.exists(os.path.join(aikit, rel)):
        return rel
    base = os.path.basename(rel)
    for pkg in KERNEL_PKGS:
        if os.path.exists(os.path.join(aikit, pkg, base)):
            return pkg + "/" + base
    return rel  # gone: the test reports it, or it was never built here


def fix_cond(path, cond):
    if path.startswith("embed/"):
        return cond  # embed's own hasAVX2
    if path.startswith("internal/cpufeat/"):
        return FLAG_RE.sub(lambda m: "Has" + m.group(1), cond)
    return FLAG_RE.sub(lambda m: "cpufeat.Has" + m.group(1), cond)


def fix_key(aikit, key):
    # file|function|condition|verdict, where the condition may hold "||".
    parts = key.split("|")
    if len(parts) < 4 or not parts[0].endswith(".go"):
        return key
    path, fn, verdict = parts[0], parts[1], parts[-1]
    cond = "|".join(parts[2:-1])
    path2 = new_path(aikit, path)
    if path2.startswith("internal/cpufeat/") and fn == "disableTopTier":
        fn = "DisableTopTier"
    return "|".join([path2, fn, fix_cond(path2, cond), verdict])


def func_pkg(aikit, fn):
    """The kernel package that now declares function fn, if it left linalg."""
    pat = re.compile(r"^func " + re.escape(fn) + r"\(", re.M)
    for pkg in KERNEL_PKGS:
        d = os.path.join(aikit, pkg)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".go") and not f.endswith("_test.go") and pat.search(open(os.path.join(d, f)).read()):
                return pkg
    return None


def fix_k3(aikit, src):
    def key(m):
        return '"' + fix_key(aikit, m.group(1)) + '"'

    out = re.sub(r'"((?:linalg|embed)/[^"|]+\.go\|[^"]*)"', key, src)

    def uncalled(m):
        fn = m.group(1)
        pkg = func_pkg(aikit, fn)
        return '"' + (pkg or "linalg") + "." + fn + '":'

    return re.sub(r'"linalg\.(\w+)":', uncalled, out)


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    check = "--check" in sys.argv[1:]
    if len(args) != 1:
        sys.exit("usage: ls3_census_fix.py [--check] AIKIT_DIR")
    aikit = args[0]
    changed = []
    for path, fix in [(CENSUS, fix_json), (K3, lambda s: fix_k3(aikit, s))]:
        src = open(path).read()
        new = fix(src)
        if new != src:
            changed.append(path)
            if not check:
                open(path, "w").write(new)
    for p in changed:
        print(("would change " if check else "changed ") + p)
    if check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
