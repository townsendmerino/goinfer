#!/usr/bin/env python3
"""Rewrite Go sources for aikit's linalg surface change (task-aikit-linalg-surface-2026-11, LS2/LS4).

Each rules file names one destination package and lists `gofmt -r` rules of the form `linalg.Name -> pkg.Name`. For every
Go file under the given roots that imports aikit's linalg:

  1. the rules whose `linalg.Name` the file uses are applied with `gofmt -r -w`, one rule per invocation;
  2. the qualifier is the destination's package name, or `aikit<name>` when the file already has an identifier of that
     name (goinfer uses `elem` as a local in places), so the rewrite cannot be shadowed;
  3. the destination's import is added beside linalg's, and linalg's import is dropped when no `linalg.` selector is
     left; then gofmt.

Comments and string literals are not rewritten (gofmt -r matches expressions only): kernelprint's labels keep the
`linalg.` names on purpose. The script prints every file it changed and every comment that still names a moved
symbol, which is the list of what a human may want to touch.

Usage: python3 scripts/aikit_surface/aikit_surface_rewrite.py [--check] RULES... -- ROOT...
  --check  report what would change, write nothing (exit 1 if anything would).
Re-runnable: a second run on its own output changes nothing.
"""

import os
import re
import subprocess
import sys

LINALG = "github.com/townsendmerino/aikit/linalg"
SKIP_DIRS = {"testdata", "vendor", ".git", "node_modules"}


def load_rules(path):
    pkg = imp = None
    rules = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("package "):
            _, pkg, imp = ln.split()
            continue
        m = re.fullmatch(r"linalg\.(\w+)\s*->\s*(\w+)\.(\w+)", ln)
        if not m or m.group(2) != pkg:
            sys.exit(f"{path}: bad rule {ln!r} (want linalg.Name -> {pkg}.Name)")
        rules.append((m.group(1), m.group(3)))
    if not pkg:
        sys.exit(f"{path}: no 'package <name> <import path>' line")
    return pkg, imp, rules


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


def imports_linalg(src):
    return re.search(r'^\s*(?:import\s+)?"' + re.escape(LINALG) + r'"\s*$', src, re.M) is not None


def add_import(src, qual, pkg, imp):
    spec = f'"{imp}"' if qual == pkg else f'{qual} "{imp}"'
    if re.search(r'^\s*(?:\w+\s+)?"' + re.escape(imp) + r'"\s*$', src, re.M):
        return src
    m = re.search(r'^(\s*)"' + re.escape(LINALG) + r'"\s*$', src, re.M)
    if m and not src[: m.start()].rstrip().endswith("import"):
        return src[: m.end()] + "\n" + m.group(1) + spec + src[m.end():]
    m = re.search(r'^import\s+"' + re.escape(LINALG) + r'"\s*$', src, re.M)
    return src[: m.start()] + f'import (\n\t"{LINALG}"\n\t{spec}\n)' + src[m.end():]


def drop_linalg_import(src):
    if re.search(r"\blinalg\.", code_only(src)):
        return src
    src = re.sub(r'^import\s+"' + re.escape(LINALG) + r'"\s*\n', "", src, flags=re.M)
    return re.sub(r'^\s*"' + re.escape(LINALG) + r'"\s*\n', "", src, flags=re.M)


def main(argv):
    check = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    if "--" not in argv:
        sys.exit(__doc__)
    i = argv.index("--")
    rulesets = [load_rules(p) for p in argv[:i]]
    roots = argv[i + 1:]
    changed, stale = [], []
    for f in go_files(roots):
        src = open(f).read()
        if not imports_linalg(src):
            continue
        out = src
        for pkg, imp, rules in rulesets:
            used = [(old, new) for old, new in rules if re.search(r"\blinalg\." + old + r"\b", code_only(out))]
            if not used:
                continue
            if imp == LINALG:
                qual = "linalg"  # a rename inside linalg: the file already imports it under that name
            else:
                qual = pkg if not re.search(r"\b" + pkg + r"\b", code_only(out)) else "aikit" + pkg
            if check:
                out = out + "\n"  # mark as changing
                continue
            open(f, "w").write(out)
            for old, new in used:
                subprocess.run(["gofmt", "-r", f"linalg.{old} -> {qual}.{new}", "-w", f], check=True)
            out = add_import(open(f).read(), qual, pkg, imp)
        if out == src:
            continue
        changed.append(f)
        if check:
            continue
        out = drop_linalg_import(out)
        open(f, "w").write(out)
        subprocess.run(["gofmt", "-w", f], check=True)
        final = open(f).read()
        for pkg, imp, rules in rulesets:
            for old, _ in rules:
                for n, ln in enumerate(final.splitlines(), 1):
                    if re.search(r"\blinalg\." + old + r"\b", ln) and "//" in ln and ln.index("//") < ln.find("linalg." + old):
                        stale.append(f"{f}:{n}: comment names linalg.{old}")
    for f in changed:
        print(("would change " if check else "changed ") + f)
    for s in stale:
        print("left for a human: " + s)
    print(f"aikit_surface_rewrite: {len(changed)} files {'would change' if check else 'changed'}")
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
