#!/usr/bin/env python3
"""The day loop: check what the diff touches, in seconds (CLAUDE.md, "Day loop").

    python3 scripts/touched.py            # the diff against the merge-base with origin/main, committed or not
    python3 scripts/touched.py -run 'TestX|TestY' decoder    # force -run for a package (needed for a slow package with no test file beside the change)
    python3 scripts/touched.py --wait 600                   # wait up to 600 s for a timed run to release the timing lock
    python3 scripts/touched.py --ignore-timing-lock         # run beside a timed run anyway (emergencies; it contaminates that run)

For each touched .go file: gofmt -l; per touched module: build and vet (the tagged variants the module needs); per touched package: go test,
with -count=1 unless internal/layout/test_cache.json marks the package cacheable (no hidden input: PW5), so Go's test cache replays it. A package whose
cold test run is over a minute (SLOW) runs only the tests declared in the touched test files and in the test file beside each touched source file (foo.go -> foo_test.go),
or the -run you name. A source-scan test (scripts/touched_scans.py: a census, a switch's case labels, every route's guard) runs
whenever a file it reads changes, from any package or module and for any kind of file, .cu and docs included.
Nothing here is a gate: CI runs the whole of it on push, and the night queue runs the heavy tier.
"""
import json, os, re, subprocess, sys, time

ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import timing_lock  # noqa: E402  (a build and test run loads the box, so it must not run under a timed one: TE9)
import touched_scans  # noqa: E402
TEST_CACHE = os.path.join(ROOT, "internal", "layout", "test_cache.json")  # PW5: which packages Go's test cache may replay
SLOW = {"decoder", "internal/sampling", "cuda", "internal/serveapp"}  # packages whose cold `go test` takes minutes; their ten-second tests call slowTest and are skipped by -short
TAGS = {"cuda": "cuda goinfer_testhooks", "gpu": "gpu goinfer_testhooks", "": "goinfer_testhooks"}


def sh(*a, **k):
    return subprocess.run(a, text=True, capture_output=True, **k)


def step(name, fn):
    t = time.time()
    ok, out = fn()
    replayed = ", cached" if ok and "(cached)" in out else ""  # go test replayed the result from its test cache
    print(f"  {'ok  ' if ok else 'FAIL'} {name} ({time.time() - t:.1f}s{replayed})")
    if not ok:
        print("".join("      " + l + "\n" for l in out.strip().splitlines()[-25:]))
    return ok


def cacheable_packages(path=TEST_CACHE):
    """The packages (repo-relative directories) whose go test may replay from Go's test cache: those test_cache.json marks
    cacheable and declares no hidden input for (internal/layout's TestTestCache_classified keeps it true). A missing or
    unreadable file makes none cacheable, so every package keeps -count=1."""
    try:
        with open(path) as fh:
            data = json.load(fh)
        declared = data.get("declared") or {}
        return {d for d, e in data["packages"].items() if e.get("cacheable") is True and d not in declared}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return set()


def count_flag(d, cacheable):
    """-count=1 for a package with a hidden input (or not known to have none), so Go's test cache cannot replay it."""
    return [] if (d or ".") in cacheable else ["-count=1"]


def module_of(d):
    """The module root (a repo-relative dir, "" for the root module) that contains dir d."""
    while d and not os.path.exists(os.path.join(ROOT, d, "go.mod")):
        d = os.path.dirname(d)
    return d


def workspace_modules():
    """The modules the active go.work uses (repo-relative, "" for the root), or None when no workspace is in use."""
    gw = sh("go", "env", "GOWORK").stdout.strip()
    if not gw or gw == "off" or not os.path.exists(gw):
        return None
    with open(gw) as fh:
        use = re.findall(r"^\s*(\.[^\s)]*)\s*$", fh.read(), re.M)
    rels = {os.path.relpath(os.path.normpath(os.path.join(os.path.dirname(gw), u)), ROOT) for u in use}
    return {"" if r == "." else r for r in rels}


def plan(changed, run_for=None):
    """What to test for a set of changed paths: {(module, package dir): -run regex, or None for the whole package}, and the
    packages skipped with why. A package in SLOW runs the tests beside its touched files; every package also runs the
    source-scan tests (touched_scans.select) that read any changed path."""
    run_for = run_for or {}
    tests, skipped = {}, []
    gofiles = sorted(f for f in changed if f.endswith(".go") and os.path.exists(os.path.join(ROOT, f)))
    bydir = {}
    for f in gofiles:
        bydir.setdefault(os.path.dirname(f), []).append(f)
    scans = touched_scans.select(changed)
    for d in sorted(set(bydir) | set(scans)):
        name = d or "."
        key = (module_of(d), d)
        if name in run_for:
            tests[key] = run_for[name]
            continue
        names = set(scans.get(d, []))
        if d in bydir and name not in SLOW:
            tests[key] = None  # the whole package already runs every scan test in it
            continue
        for f in bydir.get(d, []):
            tf = f if f.endswith("_test.go") else f[:-3] + "_test.go"
            if os.path.exists(os.path.join(ROOT, tf)):
                with open(os.path.join(ROOT, tf)) as fh:
                    names |= set(re.findall(r"^func (Test\w+)\(", fh.read(), re.M))
        if names:
            tests[key] = "^(" + "|".join(sorted(names)) + ")$"
        else:
            skipped.append(name)
    return tests, skipped


def main():
    args, run_for, wait, ignore = sys.argv[1:], {}, 0.0, False
    while args and args[0] in ("--wait", "--ignore-timing-lock"):
        if args[0] == "--wait":
            wait, args = float(args[1]), args[2:]
        else:
            ignore, args = True, args[1:]
    lock, rc = timing_lock.day_hold("touched.py", wait_s=wait, ignore=ignore)
    if rc != 0:
        return rc
    try:
        return run(args, run_for)
    finally:
        if lock:
            lock.release()


def run(args, run_for):
    """touched.py's work, run while main holds the timing lock shared."""
    while args and args[0] == "-run":
        run_for[args[2]] = args[1]
        args = args[3:]
    base = os.environ.get("TOUCHED_BASE") or sh("git", "merge-base", "HEAD", "origin/main").stdout.strip() or "HEAD~1"
    changed = set(sh("git", "diff", "--name-only", base).stdout.split()) | set(sh("git", "ls-files", "--others", "--exclude-standard").stdout.split())
    gofiles = sorted(f for f in changed if f.endswith(".go") and os.path.exists(f))
    tests, skipped = plan(changed, run_for)
    if not gofiles and not tests:
        print("touched: no .go file in the diff, and no source-scan test reads a changed file")
        return 0
    mods = {}
    for f in gofiles:
        mods.setdefault(module_of(os.path.dirname(f)), []).append(f)
    for mod, _ in tests:
        mods.setdefault(mod, [])
    print(f"touched: {len(gofiles)} .go file(s), {len(changed)} changed path(s), {len(mods)} module(s), base {base[:8]}")
    bad = 0
    if gofiles and not step("gofmt", lambda: (lambda r: (not r.stdout.strip(), r.stdout))(sh("gofmt", "-l", *gofiles))):
        bad += 1
    for name in skipped:
        print(f"  skip test {name}: slow package, no test file beside the change (name tests: touched.py -run REGEX {name})")
    ws = workspace_modules()
    cacheable = cacheable_packages()
    for mod, files in sorted(mods.items()):
        label = mod or "root"
        if ws is not None and mod not in ws:
            print(f"  skip {label}: not in go.work ({sh('go', 'env', 'GOWORK').stdout.strip()}); its tests here did not run: "
                  + ", ".join(sorted(sel or d for (m, d), sel in tests.items() if m == mod)))
            continue
        tag = TAGS.get(mod, "")
        tagarg = ["-tags", tag] if tag else []
        if mod == "metal" and sys.platform != "darwin":
            print("  skip metal: builds on darwin only")
            continue
        env = dict(os.environ, CGO_ENABLED="0") if mod == "cuda" else dict(os.environ)
        mrun = lambda *c: (lambda r: (r.returncode == 0, r.stdout + r.stderr))(sh(*c, cwd=mod or ".", env=env))
        pkgs = sorted({os.path.dirname(f) for f in files})
        rel = lambda d: "./" + os.path.relpath(d, mod or ".") if d != (mod or "") and d else "."
        if files:
            bad += not step(f"build {label}", lambda: mrun("go", "build", *tagarg, "./..."))
            bad += not step(f"vet {label} ({len(pkgs)} pkg)", lambda: mrun("go", "vet", *tagarg, *[rel(d) for d in pkgs]))
        for (m, d), sel in sorted(tests.items()):
            if m != mod:
                continue
            name = d or "."
            runarg = (["-run", sel] if sel else []) + (["-short"] if name in SLOW else [])
            bad += not step(f"test {name}" + (f" -run {sel[:40]}" if sel else ""), lambda: mrun("go", "test", *tagarg, *count_flag(d, cacheable), *runarg, rel(d)))
    print("touched: " + ("RED" if bad else "green") + " (CI runs the rest on push)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
