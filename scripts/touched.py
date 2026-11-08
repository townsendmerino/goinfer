#!/usr/bin/env python3
"""The day loop: check what the diff touches, in seconds (CLAUDE.md, "Day loop").

    python3 scripts/touched.py            # the diff against the merge-base with origin/main, committed or not
    python3 scripts/touched.py -run 'TestX|TestY' decoder    # force -run for a package (needed for a slow package with no test file beside the change)
    python3 scripts/touched.py --wait 600                   # wait up to 600 s for a timed run to release the timing lock
    python3 scripts/touched.py --ignore-timing-lock         # run beside a timed run anyway (emergencies; it contaminates that run)

For each touched .go file: gofmt -l; per touched module: build and vet (the tagged variants the module needs); per touched package: go test. A package whose
cold test run is over a minute (SLOW) runs only the tests declared in the touched test files and in the test file beside each touched source file (foo.go -> foo_test.go),
or the -run you name. A source-scan test (scripts/touched_scans.py: a census, a switch's case labels, every route's guard) runs
whenever a file it reads changes, from any package or module and for any kind of file, .cu and docs included.
Nothing here is a gate: CI runs the whole of it on push, and the night queue runs the heavy tier.

The last line is always a VERDICT line, so `| tail -1` reads the outcome: PASS (exit 0), FAIL (1), n/a when nothing in the diff has a
check (0), INCONCLUSIVE when a timed run holds the timing lock (75) or when every build, vet and test the diff needed was skipped, so
only gofmt or nothing ran (2).
"""
import json, os, re, subprocess, sys, time

ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import timing_lock  # noqa: E402  (a build and test run loads the box, so it must not run under a timed one: TE9)
import touched_scans  # noqa: E402
SLOW = {"decoder", "internal/runtime/sampling", "backend/cuda", "internal/app/serveapp"}  # packages whose cold `go test` takes minutes; their ten-second tests call slowTest and are skipped by -short
TAGS = {"backend/cuda": "cuda goinfer_testhooks", "backend/webgpu": "gpu goinfer_testhooks", "": "goinfer_testhooks"}


def sh(*a, **k):
    return subprocess.run(a, text=True, capture_output=True, **k)


def verdict(word, msg):
    return f"VERDICT: {word} — touched: {msg}"


def summary(bad, ran, skips):
    """The closing lines (a headline, then the VERDICT line) and exit code for a run of the steps named in ran, bad of them
    failed, with skips (what did not run, by name) listed. A run in which no build, vet or test ran judged nothing, so it is
    INCONCLUSIVE, not PASS, and its headline does not say green."""
    skipped = f"; did not run: {', '.join(skips)}" if skips else ""
    if bad:
        return "touched: RED (CI runs the rest on push)", verdict("FAIL", f"{bad} of {len(ran)} checks failed{skipped}"), 1
    if not [n for n in ran if not n.startswith("gofmt")]:
        return ("touched: NOTHING JUDGED (every build, vet and test was skipped)",
                verdict("INCONCLUSIVE", f"no build, vet or test ran{skipped}"), 2)
    return ("touched: green (CI runs the rest on push)",
            verdict("PASS", f"{len(ran)} checks passed (CI runs the rest on push){skipped}"), 0)


RAN = []  # the name of each step run, for the VERDICT line's count


EXCLUDED = "build constraints exclude all Go files"


def classify(mod, args, targs, env):
    """Each package of args (paths relative to module mod) for this target, keyed by its normalised relative path: "empty" when go list
    reports that build constraints exclude every file (go vet and go test both fail on such a package, so the caller skips it by name),
    "error: <msg>" for any other go list error (the caller fails it), "notests" when it builds but has no test file for the target (go
    test would pass vacuously), else "ok"."""
    if not args:
        return {}
    r = sh("go", "list", "-e", "-json=Dir,Error,GoFiles,CgoFiles,TestGoFiles,XTestGoFiles", *targs, *args, cwd=mod or ".", env=env)
    root = os.path.realpath(mod or ".")
    out, dec, buf = {}, json.JSONDecoder(), r.stdout.strip()
    while buf:
        p, n = dec.raw_decode(buf)
        buf = buf[n:].lstrip()
        key = os.path.normpath(os.path.relpath(os.path.realpath(p.get("Dir", "")), root))
        err = (p.get("Error") or {}).get("Err", "")
        if err and EXCLUDED in err:
            out[key] = "empty"
        elif err:
            out[key] = "error: " + err.strip().splitlines()[0]
        elif not p.get("TestGoFiles") and not p.get("XTestGoFiles"):
            out[key] = "notests"
        else:
            out[key] = "ok"
    if r.returncode != 0 and not out:
        out = {os.path.normpath(a): "error: " + (r.stderr.strip().splitlines() or ["go list failed"])[0] for a in args}
    return out


def step(name, fn):
    RAN.append(name)
    t = time.time()
    ok, out = fn()
    print(f"  {'ok  ' if ok else 'FAIL'} {name} ({time.time() - t:.1f}s)")
    if not ok:
        print("".join("      " + l + "\n" for l in out.strip().splitlines()[-25:]))
    return ok


NO_TESTS = "[no tests to run]"


def test_step(name, fn, skips):
    """step() for a go test run, except that a run go test reports as "[no tests to run]" (a -run that selects only tests in files the
    target excludes) is a named skip, not a pass, and does not count as a step run. Returns whether it failed."""
    t = time.time()
    ok, out = fn()
    if ok and NO_TESTS in out:
        print(f"  skip {name}: no selected test is built for this target ({time.time() - t:.1f}s)")
        skips.append(f"{name} (no selected test built)")
        return False
    RAN.append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name} ({time.time() - t:.1f}s)")
    if not ok:
        print("".join("      " + l + "\n" for l in out.strip().splitlines()[-25:]))
    return not ok


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
        print(verdict("INCONCLUSIVE", "a timed run holds the timing lock (named above); nothing ran"))
        return rc
    try:
        return run(args, run_for)
    finally:
        if lock:
            lock.release()


def check_module(mod, label, files, tests, target_os, skips, host_os=None):
    """Builds, vets and tests one module of the diff for target_os (files: its changed .go files; tests: the (module, package) -> -run
    selection the diff picked), appending what it skips to skips. Returns the number of failed steps. When target_os is not host_os (the
    go toolchain's GOHOSTOS by default), it builds and vets only: a cross-compiled test binary cannot execute here, so each test is a
    named skip."""
    host_os = host_os or sh("go", "env", "GOHOSTOS").stdout.strip()
    bad = 0
    tag = TAGS.get(mod, "")
    if mod == "backend/metal" and target_os != "darwin":
        print("  skip metal: builds on darwin only")
        skips.append("metal")
        return 0
    # The CUDA backend's devices exist only in linux files. Off linux, its tagged build cannot compile for the host, so the host gets an
    # untagged build, vet and test (what a darwin checkout compiles: the portable source scans among the tests); the other OS's build is
    # cross-checked below either way.
    cuda_off_linux = mod == "backend/cuda" and target_os != "linux"
    tagarg = [] if cuda_off_linux or not tag else ["-tags", tag]  # built afresh for every module
    env = dict(os.environ, GOOS=target_os, **({"CGO_ENABLED": "0"} if mod == "backend/cuda" else {}))
    pkgs = sorted({os.path.dirname(f) for f in files})
    rel = lambda d: "./" + os.path.relpath(d, mod or ".") if d != (mod or "") and d else "."

    mrun = lambda *c, env=env: (lambda r: (r.returncode == 0, r.stdout + r.stderr))(sh(*c, cwd=mod or ".", env=env))

    if files:
        args = [rel(d) for d in pkgs]
        kinds = classify(mod, args, tagarg, env)
        # The module build runs only when a touched package is built for the target, so a diff whose every package is excluded judges
        # nothing here and the run stays INCONCLUSIVE rather than PASS.
        if any(kinds.get(os.path.normpath(a), "ok") != "empty" for a in args):
            bad += not step(f"build {label}", lambda: mrun("go", "build", *tagarg, "./..."))
        else:
            print(f"  skip build {label}: every touched package is excluded for {target_os}")
            skips.append(f"build {label} (no files for {target_os})")
        live = []
        for a in args:
            k = kinds.get(os.path.normpath(a), "ok")
            if k == "empty":
                print(f"  skip vet {label}/{os.path.normpath(a)}: build constraints exclude all its files for {target_os}" + (" untagged" if cuda_off_linux else ""))
                skips.append(f"vet {label}/{os.path.normpath(a)} (no files for {target_os})")
            elif k.startswith("error: "):
                bad += not step(f"list {label}/{os.path.normpath(a)}", lambda k=k: (False, k))
            else:
                live.append(a)
        if live:
            bad += not step(f"vet {label} ({len(live)} pkg)", lambda: mrun("go", "vet", *tagarg, *live))
        if mod == "backend/cuda":
            # The other OS's build of the CUDA backend, module-wide (go vet type-checks _test.go files too; go build does not). Off linux:
            # the CUDA implementation, GOOS=linux at its tags. On linux: what a darwin checkout compiles, untagged. Neither needs cgo.
            if cuda_off_linux:
                other, otag, oname = dict(env, GOOS="linux", GOARCH="amd64"), ["-tags", tag], f"GOOS=linux, -tags {tag}"
                bad += not step(f"build {label} ({oname})", lambda: mrun("go", "build", *otag, "./...", env=other))
            else:
                other, otag, oname = dict(env, GOOS="darwin", GOARCH="arm64"), [], "GOOS=darwin, untagged"
            bad += not step(f"vet {label} ({oname}, ./...)", lambda: mrun("go", "vet", *otag, "./...", env=other))
    for (m, d), sel in sorted(tests.items()):
        if m != mod:
            continue
        name = d or "."
        k = classify(mod, [rel(d)], tagarg, env).get(os.path.normpath(rel(d)), "ok")
        if k == "empty":
            print(f"  skip test {name}: build constraints exclude all its files for {target_os}")
            skips.append(f"test {name} (no files for {target_os})")
            continue
        if k == "notests":
            print(f"  skip test {name}: no test file for {target_os}")
            skips.append(f"test {name} (no test files for {target_os})")
            continue
        if k.startswith("error: "):
            bad += not step(f"list {name}", lambda k=k: (False, k))
            continue
        if target_os != host_os:
            print(f"  skip test {name}: GOOS={target_os} on a {host_os} host builds and vets only (a {target_os} test binary cannot run here)")
            skips.append(f"test {name} (cross target {target_os})")
            continue
        runarg = (["-run", sel] if sel else []) + (["-short"] if name in SLOW else [])
        bad += test_step(f"test {name}" + (f" -run {sel[:40]}" if sel else ""), lambda: mrun("go", "test", *tagarg, "-count=1", *runarg, rel(d)), skips)
    return bad


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
        print(verdict("n/a", "no .go file in the diff, and no source-scan test reads a changed file"))
        return 0
    mods = {}
    for f in gofiles:
        mods.setdefault(module_of(os.path.dirname(f)), []).append(f)
    for mod, _ in tests:
        mods.setdefault(mod, [])
    print(f"touched: {len(gofiles)} .go file(s), {len(changed)} changed path(s), {len(mods)} module(s), base {base[:8]}")
    bad, skips = 0, []
    RAN.clear()
    if gofiles and not step("gofmt", lambda: (lambda r: (not r.stdout.strip(), r.stdout))(sh("gofmt", "-l", *gofiles))):
        bad += 1
    for name in skipped:
        print(f"  skip test {name}: slow package, no test file beside the change (name tests: touched.py -run REGEX {name})")
        skips.append(f"test {name}")
    ws = workspace_modules()
    target_os = sh("go", "env", "GOOS").stdout.strip() or sys.platform  # GOOS=darwin from another host builds and vets darwin; its tests do not run
    for mod, files in sorted(mods.items()):
        label = mod or "root"
        if ws is not None and mod not in ws:
            print(f"  skip {label}: not in go.work ({sh('go', 'env', 'GOWORK').stdout.strip()}); its tests here did not run: "
                  + ", ".join(sorted(sel or d for (m, d), sel in tests.items() if m == mod)))
            skips.append(label)
            continue
        bad += check_module(mod, label, files, tests, target_os, skips)
    head, line, rc = summary(bad, RAN, skips)
    print(head)
    print(line)
    return rc


if __name__ == "__main__":
    sys.exit(main())
