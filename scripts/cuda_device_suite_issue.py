#!/usr/bin/env python3
"""Turn the cuda-device-suite verdict into a GitHub issue: red opens one (or comments on the open one) with the failing names and the log tail; green closes
the open one with a comment. The issue is titled "cuda-device-suite red on <short sha>". Called by the night job after scripts/cuda_device_suite.py has
written its log; `--gh` names the gh binary so a test can pass a fake.

  python3 scripts/cuda_device_suite_issue.py --verdict-log LOG --sha abc1234 [--repo townsendmerino/goinfer] [--gh gh]

Exit 0 when the issue state now matches the verdict, 2 when the log carries no verdict line (nothing is opened or closed: a job that never reached a verdict is
in night.py's SUMMARY.md, and is not a red test), 1 when a gh call failed.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile

TITLE = "cuda-device-suite red on "
TAIL_LINES = 40


def parse_verdict(lines):
    """'RED' or 'GREEN' from the last VERDICT line of the runner's output; None when there is none."""
    for l in reversed(lines):
        m = re.match(r"VERDICT: (RED|GREEN)\s*$", l)
        if m:
            return m.group(1)
    return None


def names_block(lines):
    """The runner's summary and name lists: from its 'cuda-device-suite: N passed' line to the verdict."""
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("cuda-device-suite: "):
            return lines[i:]
    return []


def scrub(text):
    return text.replace("/home/francis", "~")


class Gh:
    def __init__(self, gh, repo):
        self.gh, self.repo = gh, repo

    def run(self, *args):
        p = subprocess.run([self.gh, *args, "--repo", self.repo], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"gh {' '.join(args[:2])} failed: {p.stderr.strip()[:300]}")
        return p.stdout

    def open_issues(self):
        out = self.run("issue", "list", "--state", "open", "--search", f'"{TITLE.strip()}" in:title', "--json", "number,title", "--limit", "30")
        return sorted((i["number"] for i in json.loads(out or "[]") if i["title"].startswith(TITLE)), reverse=True)

    def body_file(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        f.write(text)
        f.close()
        return f.name


def body_for(sha, lines):
    names = names_block(lines) or ["(the runner printed no summary)"]
    tail = lines[-TAIL_LINES:]
    return scrub(f"Revision `{sha}` of `origin/main`, run by the night job `cuda-device-suite` on nobara with the device present "
                 f"(docs/measurements/cuda-device-suite-2026-10-18/README.md).\n\n```\n" + "\n".join(names) + "\n```\n\n"
                 f"Last {len(tail)} lines of the log:\n\n```\n" + "\n".join(tail) + "\n```\n")


def report(verdict, sha, lines, gh):
    open_now = gh.open_issues()
    if verdict == "RED":
        body = gh.body_file(body_for(sha, lines))
        if open_now:
            gh.run("issue", "comment", str(open_now[0]), "--body-file", body)
            return f"commented on #{open_now[0]}"
        gh.run("issue", "create", "--title", TITLE + sha, "--body-file", body)
        return "opened an issue"
    done = []
    for n in open_now:
        gh.run("issue", "close", str(n), "--comment", f"cuda-device-suite is green on `{sha}`.")
        done.append(f"#{n}")
    return "closed " + ", ".join(done) if done else "green, nothing open"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict-log", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--repo", default="townsendmerino/goinfer")
    ap.add_argument("--gh", default="gh")
    a = ap.parse_args(argv)
    lines = open(a.verdict_log, errors="replace").read().splitlines()
    verdict = parse_verdict(lines)
    if verdict is None:
        print("no VERDICT line in the log; no issue opened or closed")
        return 2
    try:
        print(f"{verdict}: {report(verdict, a.sha, lines, Gh(a.gh, a.repo))}")
    except RuntimeError as e:
        print(f"issue update failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
