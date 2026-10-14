#!/usr/bin/env python3
"""The day loop: check what the diff touches, in seconds (CLAUDE.md, "Day loop").

    python3 scripts/touched.py            # the diff against the merge-base with origin/main, committed or not
    python3 scripts/touched.py -run 'TestX|TestY' decoder    # force -run for a package (needed for a slow package with no test file beside the change)
    python3 scripts/touched.py --wait 600                   # wait up to 600 s for a timed run to release the timing lock
    python3 scripts/touched.py --ignore-timing-lock         # run beside a timed run anyway (emergencies; it contaminates that run)

For each touched .go file: gofmt -l; per touched module: build and vet (the tagged variants the module needs); per touched package: go test. A package whose
cold test run is over a minute (SLOW) runs only the tests declared in the touched test files and in the test file beside each touched source file (foo.go -> foo_test.go),
or the -run you name. Nothing here is a gate: CI runs the whole of it on push, and the night queue runs the heavy tier.
"""
import os, re, subprocess, sys, time

ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import timing_lock  # noqa: E402  (a build and test run loads the box, so it must not run under a timed one: TE9)
SLOW = {"decoder", "internal/sampling", "cuda"}  # packages whose cold `go test` takes minutes; their ten-second tests call slowTest and are skipped by -short
TAGS = {"cuda": "cuda goinfer_testhooks", "gpu": "gpu goinfer_testhooks", "": "goinfer_testhooks"}


def sh(*a, **k):
    return subprocess.run(a, text=True, capture_output=True, **k)


def step(name, fn):
    t = time.time()
    ok, out = fn()
    print(f"  {'ok  ' if ok else 'FAIL'} {name} ({time.time() - t:.1f}s)")
    if not ok:
        print("".join("      " + l + "\n" for l in out.strip().splitlines()[-25:]))
    return ok


def main():
    args, run_for, wait, ignore = sys.argv[1:], {}, 0.0, False
    while args and args[0] in ("--wait", "--ignore-timing-lock"):
        if args[0] == "--wait":
            wait, args = float(args[1]), args[2:]
        else:
            ignore, args = True, args[1:]
    if (rc := timing_lock.day_check("touched.py", wait_s=wait, ignore=ignore)) != 0:
        return rc
    while args and args[0] == "-run":
        run_for[args[2]] = args[1]
        args = args[3:]
    base = os.environ.get("TOUCHED_BASE") or sh("git", "merge-base", "HEAD", "origin/main").stdout.strip() or "HEAD~1"
    changed = set(sh("git", "diff", "--name-only", base).stdout.split()) | set(sh("git", "ls-files", "--others", "--exclude-standard").stdout.split())
    gofiles = sorted(f for f in changed if f.endswith(".go") and os.path.exists(f))
    if not gofiles:
        print("touched: no .go file in the diff")
        return 0
    mods = {}
    for f in gofiles:
        d = os.path.dirname(f)
        while d and not os.path.exists(os.path.join(d, "go.mod")):
            d = os.path.dirname(d)
        mods.setdefault(d, []).append(f)
    print(f"touched: {len(gofiles)} .go file(s) in {len(mods)} module(s), base {base[:8]}")
    bad = 0
    if not step("gofmt", lambda: (lambda r: (not r.stdout.strip(), r.stdout))(sh("gofmt", "-l", *gofiles))):
        bad += 1
    for mod, files in sorted(mods.items()):
        label = mod or "root"
        tag = TAGS.get(mod, "")
        tagarg = ["-tags", tag] if tag else []
        if mod == "metal" and sys.platform != "darwin":
            print("  skip metal: builds on darwin only")
            continue
        env = dict(os.environ, CGO_ENABLED="0") if mod == "cuda" else dict(os.environ)
        mrun = lambda *c: (lambda r: (r.returncode == 0, r.stdout + r.stderr))(sh(*c, cwd=mod or ".", env=env))
        bydir = {}
        for f in files:
            bydir.setdefault(os.path.dirname(f), []).append(f)
        pkgs = {d: ("./" + os.path.relpath(d, mod or ".") if d != (mod or "") and d else ".") for d in bydir}
        bad += not step(f"build {label}", lambda: mrun("go", "build", *tagarg, "./..."))
        bad += not step(f"vet {label} ({len(pkgs)} pkg)", lambda: mrun("go", "vet", *tagarg, *sorted(pkgs.values())))
        for d, p in sorted(pkgs.items()):
            name = d or "."
            sel = run_for.get(name)
            if not sel and name in SLOW:
                tests = set()
                for f in bydir[d]:
                    tf = f if f.endswith("_test.go") else f[:-3] + "_test.go"
                    if os.path.exists(tf):
                        tests |= set(re.findall(r"^func (Test\w+)\(", open(tf).read(), re.M))
                sel = "^(" + "|".join(sorted(tests)) + ")$" if tests else None
                if not sel:
                    print(f"  skip test {name}: slow package, no test file beside the change (name tests: touched.py -run REGEX {name})")
                    continue
            runarg = (["-run", sel] if sel else []) + (["-short"] if name in SLOW else [])
            bad += not step(f"test {name}" + (f" -run {sel[:40]}" if sel else ""), lambda: mrun("go", "test", *tagarg, "-count=1", *runarg, p))
    print("touched: " + ("RED" if bad else "green") + " (CI runs the rest on push)")
    return 1 if bad else 0


sys.exit(main())
