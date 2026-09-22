#!/usr/bin/env python3
"""ci_test_summary.py -- the readable half of `go test -json ./... > test.json`.

The linux `test` job in .github/workflows/ci.yml captures `go test -json` to a file instead of
printing `go test`'s plain output (docs/completed/task-ci-speed-2026-09.md, C7): the JSON is the
only source of PER-TEST timings from a real CI runner, which the CI-speed work had no way to get
and needs before ./decoder can be sliced across the runner's cores. This script turns the file
back into what a human reading the job log wants, in this order:

  1. every build failure, verbatim (Go >= 1.24 emits build output as JSON events keyed by
     ImportPath, so with stdout redirected these would otherwise be invisible);
  2. the full captured output of every failed test, verbatim;
  3. one line per package -- ok / FAIL / skipped / `(cached)` -- with elapsed seconds;
  4. the N slowest top-level tests (a subtest's time is inside its parent's Elapsed).

Exit status is 1 only when the stream holds NO package-level result at all: a capture that
recorded nothing must not read as green (the same zero-match-guard shape as the sampler gates).
Everything else exits 0 -- the `go test` step's own exit code is what fails the job; this script
reports, it does not judge.

Usage: ci_test_summary.py test.json [--top N]
"""

import json
import os
import sys
from collections import defaultdict


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = argv[1]
    top_n = 30
    if "--top" in argv:
        top_n = int(argv[argv.index("--top") + 1])

    build_out = defaultdict(list)  # ImportPath -> output lines
    build_failed = []  # ImportPaths with a build-fail event
    test_out = defaultdict(list)  # (pkg, test) -> output lines
    pkg_result = {}  # pkg -> (action, elapsed)
    pkg_cached = set()
    failed_tests = []  # (pkg, test) in order of failure
    test_elapsed = []  # (elapsed, pkg, test) for top-level tests
    non_json = []
    n_events = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                non_json.append(line)
                continue
            n_events += 1
            action = e.get("Action")
            if "ImportPath" in e:  # build events (go >= 1.24)
                ip = e["ImportPath"]
                if action == "build-output":
                    build_out[ip].append(e.get("Output", ""))
                elif action == "build-fail":
                    build_failed.append(ip)
                continue
            pkg = e.get("Package", "")
            test = e.get("Test")
            if action == "output":
                out = e.get("Output", "")
                test_out[(pkg, test)].append(out)
                if test is None and "(cached)" in out:
                    pkg_cached.add(pkg)
            elif action in ("pass", "fail", "skip"):
                if test is None:
                    pkg_result[pkg] = (action, e.get("Elapsed", 0.0))
                else:
                    if action == "fail":
                        failed_tests.append((pkg, test))
                    if "/" not in test:
                        test_elapsed.append((e.get("Elapsed", 0.0), pkg, test))

    lines = []

    if build_failed or build_out:
        lines.append("== BUILD FAILURES ==")
        for ip in build_failed or sorted(build_out):
            lines.append(f"--- {ip}")
            lines.extend(l.rstrip("\n") for l in build_out.get(ip, []))
        lines.append("")

    if failed_tests:
        lines.append(f"== FAILED TESTS ({len(failed_tests)}) ==")
        for pkg, test in failed_tests:
            lines.append(f"--- FAIL {short(pkg)} {test}")
            lines.extend(l.rstrip("\n") for l in test_out.get((pkg, test), []))
        # A package can fail without any test failing (a panic outside a test, a TestMain exit):
        # show that package's untargeted output so the reason is on the log.
        lines.append("")
    for pkg, (action, _) in sorted(pkg_result.items()):
        if action == "fail" and not any(p == pkg for p, _ in failed_tests):
            lines.append(f"== PACKAGE FAILED WITHOUT A FAILING TEST: {short(pkg)} ==")
            lines.extend(l.rstrip("\n") for l in test_out.get((pkg, None), []))
            lines.append("")

    lines.append("== PACKAGES ==")
    wall = 0.0
    total = 0.0
    for pkg in sorted(pkg_result):
        action, elapsed = pkg_result[pkg]
        tag = {"pass": "ok  ", "fail": "FAIL", "skip": "skip"}[action]
        note = " (cached)" if pkg in pkg_cached else ""
        lines.append(f"{tag} {elapsed:8.1f}s  {short(pkg)}{note}")
        total += elapsed
        wall = max(wall, elapsed)
    lines.append(
        f"{len(pkg_result)} packages, {sum(1 for p in pkg_cached)} cached; "
        f"sum of package elapsed {total:.0f}s, longest package {wall:.0f}s"
    )
    lines.append("")

    test_elapsed.sort(reverse=True)
    lines.append(f"== SLOWEST {min(top_n, len(test_elapsed))} TOP-LEVEL TESTS (of {len(test_elapsed)}) ==")
    for elapsed, pkg, test in test_elapsed[:top_n]:
        lines.append(f"{elapsed:8.1f}s  {short(pkg)}  {test}")
    lines.append("")

    if non_json:
        lines.append(f"== {len(non_json)} NON-JSON LINES (older go, or something wrote to stdout) ==")
        lines.extend(non_json[:200])
        lines.append("")

    text = "\n".join(lines)
    print(text)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as s:
            s.write("```\n" + text + "\n```\n")

    if not pkg_result:
        print(
            f"::error::{path}: {n_events} JSON events but NO package-level result -- the capture "
            "recorded nothing; this must not read as green",
            file=sys.stderr,
        )
        return 1
    return 0


def short(pkg):
    prefix = "github.com/townsendmerino/goinfer"
    if pkg == prefix:
        return "."
    if pkg.startswith(prefix + "/"):
        return pkg[len(prefix) + 1 :]
    return pkg


if __name__ == "__main__":
    sys.exit(main(sys.argv))
