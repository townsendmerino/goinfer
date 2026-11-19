#!/usr/bin/env python3
"""ci_flakes.py: a read-only record of the last day's failed ci jobs, for telling flakes from defects.

One record per failed job attempt in the `ci` workflow's runs created in the window: lane (job name), run, attempt, sha,
event, the failing tests (from root-windows' `root-windows-test-json` artifact where the run has one, otherwise the
`--- FAIL:` lines of the job's log, otherwise the log's tail before its first error), whether a later attempt of the same
job in the same run passed, and which other lanes failed on the same sha. It writes Markdown (the job summary) and JSON;
it never commits, opens issues or comments. The scheduled workflow is .github/workflows/ci-flakes.yml.

Usage: ci_flakes.py --repo <owner/name> [--hours 24] [--out DIR]
Tests: python3 -B scripts/test_ci_flakes.py
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile

WINDOWS_ARTIFACT = "root-windows-test-json"
TAIL_LINES = 15

_TS = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ")
_FAIL = re.compile(r"--- FAIL: (Test[A-Za-z0-9_]+(?:/[^\s\"\\]+)?)")


def failed_tests_from_test_json(text):
    """go test -json stream -> sorted failing tests (subtests included), plus `<package> (package)` for a package that
    failed with no failing test (a build failure or a crash)."""
    tests, pkgs_failed, pkgs_with_test = set(), set(), set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("Action") != "fail":
            continue
        pkg = ev.get("Package", "")
        if ev.get("Test"):
            tests.add(ev["Test"])
            pkgs_with_test.add(pkg)
        elif pkg:
            pkgs_failed.add(pkg)
    return sorted(tests) + sorted(f"{p} (package)" for p in pkgs_failed - pkgs_with_test)


def failed_tests_from_log(text):
    """A job log -> (failing tests from its `--- FAIL:` lines, or [] and the log's tail before its first ##[error])."""
    lines = [_TS.sub("", l) for l in text.splitlines()]
    tests = sorted({m.group(1) for l in lines for m in _FAIL.finditer(l)})
    if tests:
        return tests, ""
    end = next((i for i, l in enumerate(lines) if l.startswith("##[error]")), len(lines))
    tail = [l for l in lines[max(0, end - TAIL_LINES):end + 1] if not l.startswith("##[")]
    if end < len(lines):
        tail.append(lines[end])
    return [], "\n".join(tail).strip()


def build_records(runs, jobs, tests_for):
    """runs: workflow-run objects. jobs: {(run_id, attempt): [job objects]}. tests_for(run, job) -> (tests, tail).
    Returns one record per failed job attempt, ordered by run then lane."""
    failed_by_sha = {}
    for run in runs:
        for a in range(1, (run.get("run_attempt") or 1) + 1):
            for j in jobs.get((run["id"], a), []):
                if j.get("conclusion") == "failure":
                    failed_by_sha.setdefault(run["head_sha"], set()).add(j["name"])
    out = []
    for run in sorted(runs, key=lambda r: r["id"]):
        last = run.get("run_attempt") or 1
        for a in range(1, last + 1):
            for j in sorted(jobs.get((run["id"], a), []), key=lambda j: j["name"]):
                if j.get("conclusion") != "failure":
                    continue
                later = [x.get("conclusion") for b in range(a + 1, last + 1)
                         for x in jobs.get((run["id"], b), []) if x["name"] == j["name"]]
                tests, tail = tests_for(run, j)
                out.append({
                    "lane": j["name"],
                    "run": run["id"],
                    "attempt": a,
                    "sha": run["head_sha"],
                    "event": run.get("event"),
                    "branch": run.get("head_branch"),
                    "failed_step": next((s["name"] for s in j.get("steps") or [] if s.get("conclusion") == "failure"), None),
                    "tests": tests,
                    "log_tail": tail,
                    "rerun_passed": None if not later else ("success" in later),
                    "other_lanes_failed_on_sha": sorted(failed_by_sha.get(run["head_sha"], set()) - {j["name"]}),
                })
    return out


def markdown(records, since):
    lines = [f"### ci failures since {since}", ""]
    if not records:
        return "\n".join(lines + ["No failed ci job in the window."]) + "\n"
    lines += ["| lane | run (attempt) | sha | event | failing tests or step | re-run passed | other lanes red on sha |",
              "|---|---|---|---|---|---|---|"]
    for r in records:
        what = ", ".join(r["tests"][:5]) + (f" (+{len(r['tests']) - 5})" if len(r["tests"]) > 5 else "")
        if not what:
            last = next((l for l in reversed(r["log_tail"].splitlines()) if l and not l.startswith("##[")), "")
            what = f"step: {r['failed_step'] or '?'}" + (f": `{last[:100].replace('|', '/').replace('`', '')}`" if last else "")
        rerun = {None: "not re-run", True: "yes", False: "no"}[r["rerun_passed"]]
        others = ", ".join(r["other_lanes_failed_on_sha"]) or "none (solo)"
        lines.append(f"| {r['lane']} | {r['run']} ({r['attempt']}) | {r['sha'][:9]} | {r['event']} | {what} | {rerun} | {others} |")
    solo = [r for r in records if not r["other_lanes_failed_on_sha"]]
    lines += ["", f"{len(records)} failed job attempts, {len(solo)} solo (no other lane red on the same sha)."]
    return "\n".join(lines) + "\n"


# --- I/O: gh with the workflow token, read-only.

def _gh_json(repo, path):
    out = subprocess.run(["gh", "api", f"repos/{repo}/{path}"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _runs_since(repo, since):
    runs, page = [], 1
    while True:
        r = _gh_json(repo, f"actions/workflows/ci.yml/runs?created=%3E%3D{since}&status=completed&per_page=100&page={page}")
        batch = r.get("workflow_runs") or []
        runs += batch
        if len(batch) < 100:
            return runs
        page += 1


def _jobs(repo, run_id, attempt):
    jobs, page = [], 1
    while True:
        r = _gh_json(repo, f"actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100&page={page}")
        batch = r.get("jobs") or []
        jobs += batch
        if len(batch) < 100:
            return jobs
        page += 1


def _tests_for(repo):
    def call(run, job):
        if job["name"] == "root-windows":
            with tempfile.TemporaryDirectory() as d:
                r = subprocess.run(["gh", "run", "download", str(run["id"]), "-R", repo, "-n", WINDOWS_ARTIFACT, "-D", d],
                                   capture_output=True, text=True)
                p = os.path.join(d, "test.json")
                if r.returncode == 0 and os.path.exists(p):
                    with open(p, errors="replace") as f:
                        tests = failed_tests_from_test_json(f.read())
                    if tests:
                        return tests, ""
        r = subprocess.run(["gh", "api", f"repos/{repo}/actions/jobs/{job['id']}/logs"], capture_output=True, text=True)
        if r.returncode != 0:
            return [], f"(log unavailable: {r.stderr.strip()[:200]})"
        return failed_tests_from_log(r.stdout)
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=a.hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = _runs_since(a.repo, since)
    jobs = {(r["id"], n): _jobs(a.repo, r["id"], n) for r in runs for n in range(1, (r.get("run_attempt") or 1) + 1)}
    records = build_records(runs, jobs, _tests_for(a.repo))
    md = markdown(records, since)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ci-flakes.json"), "w") as f:
        json.dump({"since": since, "runs": len(runs), "records": records}, f, indent=1)
    with open(os.path.join(a.out, "ci-flakes.md"), "w") as f:
        f.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
