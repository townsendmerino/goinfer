#!/usr/bin/env python3
"""asset_registry.py — THE predicate for "is this test asset present", and its only implementation
outside of decoder/asset_registry_test.go.

WHY THIS EXISTS. `parity_sweep.sh` decided asset presence with `[ -e "$path" ]` while every gate
decided it again inline, with its own default path and its own check. The two answers were free to
disagree, and did:

  * a DIRECTORY satisfies `-e`, so three preflight entries named a directory where the loader wanted
    a .gguf FILE, and preflight reported them RESOLVED. Four gates were costed by that.
  * GOINFER_QWEN35_GOLDEN's real requirement is a readable manifest.json INSIDE the directory. `-e`
    on the directory cannot say that, so preflight said present and the gate skipped anyway.
  * GOINFER_PREQUANT_GGUF had three different fallbacks across four call sites and none at a fourth.

The fix is not a better `-e`. It is that presence is DEFINED IN ONE PLACE (testdata/assets.json:
kind, members, min_bytes) and every consumer applies that definition rather than approximating it.

TWO IMPLEMENTATIONS, ONE PREDICATE — AND A GATE THAT PROVES IT. Bash cannot read JSON and Go cannot
be called from the preflight, so the predicate is implemented twice: here and in
decoder/asset_registry_test.go. That is a divergence risk, so it is GATED, not trusted:
TestAssetRegistry_agreesWithPreflight asks this script for its verdict on every entry and asserts the
Go side agrees, path for path. Two implementations that are checked against each other are a
different thing from two implementations nobody compares -- which is what we had.

    preflight [--export-to FILE]   the sweep's table; writes `export VAR=path` lines for the resolved
    check --env VAR                exit 0 if resolvable (prints the path), 1 if not
    list                           registry env var names, one per line
    verdicts                       JSON {env: {path, ok, reason}} — what the Go conformance test reads
    census                         asset-shaped resolutions in _test.go that this registry does NOT
                                   cover — the denominator, so a registry covering 10 of N never
                                   reports its 10 as though it were N
"""

import argparse
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "testdata/assets.json"


def models_root():
    return pathlib.Path(os.environ.get("GOINFER_MODELS") or (pathlib.Path.home() / "models"))


def expand(cand):
    """$REPO and $MODELS are the only two roots. A candidate naming neither is a bug in the registry,
    not a relative path to guess at, so it is returned unchanged and will simply fail the predicate."""
    if cand.startswith("$REPO/"):
        return ROOT / cand[len("$REPO/"):]
    if cand.startswith("$MODELS/"):
        return models_root() / cand[len("$MODELS/"):]
    return pathlib.Path(cand)


def satisfies(a, p):
    """THE PREDICATE. Returns (ok, reason). `reason` names what was wrong, because "not found" on a
    path that exists is the single most misleading thing this can print."""
    p = pathlib.Path(p)
    kind = a.get("kind", "file")
    if not p.exists():
        return False, "does not exist"
    if kind == "file":
        if p.is_dir():
            # The bug that cost four gates, now a named outcome instead of a silent pass.
            return False, "is a DIRECTORY, but this asset is a file"
        if not p.is_file():
            return False, "exists but is not a regular file"
        n = p.stat().st_size
        if n < a.get("min_bytes", 1):
            return False, f"only {n} bytes (min_bytes {a.get('min_bytes', 1)}) — a stub or a truncated copy"
        return True, ""
    if kind == "dir":
        if not p.is_dir():
            return False, "is not a directory"
        missing = [m for m in a.get("members", []) if not (p / m).exists()]
        if missing:
            return False, f"directory exists but is missing {', '.join(missing)}"
        anyof = a.get("members_any", [])
        if anyof and not any((p / m).exists() for m in anyof):
            return False, f"directory exists but has none of {', '.join(anyof)}"
        return True, ""
    return False, f"unknown kind {kind!r} in the registry"


def resolve(a):
    """(path_or_None, source, reason). An explicit env value ALWAYS wins and is checked by the same
    predicate: falling back to a candidate when the operator named a path would silently run the gate
    against a different file than the one they asked for."""
    cur = os.environ.get(a["env"], "")
    if cur:
        ok, why = satisfies(a, cur)
        return (cur if ok else None), ("env" if ok else "env!!"), why
    tried = []
    for c in a.get("candidates", []):
        p = expand(c)
        ok, why = satisfies(a, p)
        if ok:
            return str(p), "resolved", ""
        tried.append(f"{p} ({why})")
    return None, "NOT FOUND", "; ".join(tried) if tried else "no candidates in the registry"


def load():
    return json.loads(REGISTRY.read_text())["assets"]


def cmd_preflight(a):
    assets = load()
    exports, miss = [], 0
    print(f"== asset preflight — {len(assets)} registered asset(s), models root {models_root()} ==")
    print("   predicate: testdata/assets.json (kind/members/min_bytes) — the SAME one the gates apply")
    for e in assets:
        path, src, why = resolve(e)
        shown = path if path else (os.environ.get(e["env"], "") or "—")
        print(f"   {e['env']:<28} {src:<9} {shown}")
        if path:
            exports.append(f"export {e['env']}={json.dumps(path)}")
        else:
            miss += 1
            # The reason, not just the verdict: "NOT FOUND" on a path that exists is the report that
            # sent someone to look for a missing file for forty minutes.
            for part in (why or "").split("; "):
                if part:
                    print(f"   {'':<28} {'':<9}   ^ {part}")
    # Derived, never a literal. The message this replaced said "all 10 assets resolved" while the
    # table above it had nine rows -- the constant-that-drifts, in the line announcing correctness.
    if miss:
        print(f"   ^^ {miss} of {len(assets)} UNRESOLVED — the gates needing them will skip, and a skip is")
        print("      reported as a blocker. Fix the path above rather than reading the blocker list as a")
        print("      tree defect.")
    else:
        print(f"   all {len(assets)} registered assets resolved — a blocker below is about the TREE, not"
              " the invocation")
    if a.export_to:
        pathlib.Path(a.export_to).write_text("\n".join(exports) + ("\n" if exports else ""))
    return 0


def cmd_check(a):
    for e in load():
        if e["env"] == a.env:
            path, src, why = resolve(e)
            if path:
                print(path)
                return 0
            print(f"{a.env}: {src} — {why}", file=sys.stderr)
            return 1
    print(f"{a.env}: not in the registry ({REGISTRY})", file=sys.stderr)
    return 2


def cmd_list(a):
    for e in load():
        print(e["env"])
    return 0


def cmd_verdicts(a):
    out = {}
    for e in load():
        path, src, why = resolve(e)
        out[e["env"]] = {"path": path, "ok": path is not None, "source": src, "reason": why}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


ENVPAT = re.compile(r'os\.Getenv\("(GOINFER_[A-Z0-9_]+)"\)')
# The BINDING form: `path := os.Getenv("V")`. A bare Getenv in a comparison binds nothing.
BINDPAT = re.compile(r'\b([A-Za-z_]\w*)\s*:?=\s*os\.Getenv\("(GOINFER_[A-Z0-9_]+)"\)')

# Does this variable NAME a path, or does it carry a switch or a scalar? Classifying by NAME would be
# the guess this registry exists to replace (GOINFER_MOE_GIW is a path, GOINFER_MOE_NOCACHE is not,
# and nothing in the names says so). So classify by USE, with the rule written down:
#
#   a variable is PATH-SHAPED if the identifier its value is BOUND TO subsequently flows into a
#   filesystem call or a model load, within PATH_WINDOW lines.
#
# PROXIMITY WAS THE WRONG RULE and its errors were instructive: "any filesystem call within N lines"
# called GOINFER_HEAVY_TESTS -- a pure on/off switch -- path-shaped, because the test it guards
# happens to open a file three lines later. Following the bound identifier asks the question that
# actually matters (does THIS VALUE become a path) rather than a question about layout.
#
# AND THERE IS A THIRD ANSWER, because no local rule gets this right. GOINFER_SERVE_MODEL is a path
# by any human reading, but its value flows into a struct field (`path: path`) and never touches a
# filesystem call in that function -- nothing a window can see says "path". Calling it a switch would
# be a bucket absorbing what the rule did not determine, which is the census failure this whole line
# of work exists to stop. So it lands in UNCLASSIFIED and is counted there, out loud.
#
# The remedy for a misclassification in ANY direction is the same: REGISTER the asset, which removes
# it from the heuristic's reach entirely. Do not tune the window to make a number look better -- that
# turns the denominator back into decoration.
PATH_CALL = r'os\.(?:Stat|Open|OpenFile|ReadFile|ReadDir|Lstat)|Load|LoadModel|filepath\.(?:Join|Abs|Dir|Glob)'
PATH_WINDOW = 12


def cmd_census(a):
    """The denominator. A registry that covers ten assets and reports on ten assets is the census
    shape all over again: accurate about its numerator, silent about its universe."""
    registered = {e["env"] for e in load()}
    found, verdicts = {}, {}
    for f in sorted(ROOT.rglob("*_test.go")):
        if "/testdata/" in str(f):
            continue
        lines = f.read_text(errors="replace").splitlines()
        for i, ln in enumerate(lines):
            for m in ENVPAT.finditer(ln):
                v = m.group(1)
                found.setdefault(v, set()).add(str(f.relative_to(ROOT)))
                bm = BINDPAT.search(ln)
                if not bm or bm.group(2) != v or bm.group(1) == "_":
                    # Bare `os.Getenv("V") != ""` binds nothing and IS a comparison — a switch. Bare
                    # `f(os.Getenv("V"))` binds nothing either, but says nothing about the value.
                    verdicts.setdefault(v, set()).add(
                        "switch" if re.search(r'Getenv\("' + v + r'"\)\s*[!=]=', ln) else "unclassified")
                    continue
                ident = bm.group(1)
                win = lines[i:i + PATH_WINDOW]
                flows = re.compile(r'\b(?:' + PATH_CALL + r')\([^)]*\b' + re.escape(ident) + r'\b')
                if any(flows.search(x) for x in win):
                    verdicts.setdefault(v, set()).add("path")
                    continue
                uses = [x for x in win[1:] if re.search(r'\b' + re.escape(ident) + r'\b', x)]
                cmpish = re.compile(r'\b' + re.escape(ident) + r'\b\s*[!=]=|'
                                    r'strconv\.\w+\(\s*' + re.escape(ident))
                verdicts.setdefault(v, set()).add(
                    "switch" if uses and all(cmpish.search(x) for x in uses) else "unclassified")

    def verdict(v):
        s = verdicts.get(v, {"unclassified"})
        return "path" if "path" in s else ("switch" if s == {"switch"} else "unclassified")

    # THE UNIVERSE IS found | registered, NOT found. Registering an asset REMOVES its os.Getenv from
    # the test sources (that is the whole point), so a universe of "vars read by tests" shrinks by one
    # every time this work succeeds -- and the buckets stop summing to the total. That is the
    # numerator-vs-universe defect, in the tool built to report it. Counting the union keeps the four
    # buckets a partition and keeps the total stable across conversions.
    universe = set(found) | registered
    path = sorted(v for v in universe if v not in registered and verdict(v) == "path")
    switch = sorted(v for v in universe if v not in registered and verdict(v) == "switch")
    unk = sorted(v for v in universe if v not in registered and verdict(v) == "unclassified")

    print(f"asset registry census — {len(universe)} GOINFER_* asset-candidate vars "
          f"(read by a test, or registered, or both): {len(registered)} REGISTERED, {len(path)} "
          f"PATH-SHAPED unregistered, {len(switch)} switch/scalar, {len(unk)} UNCLASSIFIED")
    assert len(registered) + len(path) + len(switch) + len(unk) == len(universe), \
        "census buckets must partition the universe"
    print(f"  Classified by USE, not by name, following the identifier the value is bound to for "
          f"{PATH_WINDOW} lines (PATH_CALL in this script).")
    print("  UNCLASSIFIED is a real answer, not a leftover: the value goes somewhere this rule cannot")
    print("  follow, so whether a presence predicate applies to it is UNKNOWN rather than 'no'.")
    print()
    print("PATH-SHAPED BUT UNREGISTERED — each is a gate deciding asset presence its own way, which is")
    print("exactly the condition the registry removes for the ten it covers:")
    for k in path:
        files = sorted(found[k])
        print(f"  {k:<34} {files[0]}" + (f" (+{len(files)-1} more)" if len(files) > 1 else ""))
    print()
    print("UNCLASSIFIED — the rule could not tell. Some of these ARE assets (GOINFER_SERVE_MODEL is a")
    print(".gguf path that flows into a struct field, never into a filesystem call):")
    for k in unk:
        files = sorted(found[k])
        print(f"  {k:<34} {files[0]}" + (f" (+{len(files)-1} more)" if len(files) > 1 else ""))
    print()
    print(f"REGISTERED ({len(registered)}): {', '.join(sorted(registered))}")
    print()
    print(f"SWITCH/SCALAR ({len(switch)}): {', '.join(switch)}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("preflight")
    pf.add_argument("--export-to")
    pf.set_defaults(fn=cmd_preflight)

    ck = sub.add_parser("check")
    ck.add_argument("--env", required=True)
    ck.set_defaults(fn=cmd_check)

    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("verdicts").set_defaults(fn=cmd_verdicts)
    sub.add_parser("census").set_defaults(fn=cmd_census)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
