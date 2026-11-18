#!/usr/bin/env python3
"""Turn the cuda-device-suite verdict into a GitHub issue: red opens one (or comments on the open one) with the failing names and the log tail; green closes
the open one with a comment. The issue is titled "cuda-device-suite red on <short sha>". Called by the night job after scripts/cuda_device_suite.py has
written its log; `--gh` names the gh binary so a test can pass a fake.

  python3 scripts/cuda_device_suite_issue.py --verdict-log LOG --sha abc1234 [--repo townsendmerino/goinfer] [--gh gh]

Exit 0 when the issue state now matches the verdict, 2 when the log carries no verdict line (nothing is opened or closed: a job that never reached a verdict is
in night.py's SUMMARY.md, and is not a red test), 1 when a gh call failed.
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile

TITLE = "cuda-device-suite red on "  # the cuda title; another backend's is "<backend>-device-suite red on "
STATUS_PREFIX = "device-suite status: "
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
        if re.match(r"[a-z]+-device-suite: ", lines[i]):
            return lines[i:]
    return []


def scrub(text):
    return text.replace("/home/francis", "~")


class Gh:
    def __init__(self, gh, repo, title=TITLE):
        self.gh, self.repo, self.title = gh, repo, title

    def run(self, *args):
        p = subprocess.run([self.gh, *args, "--repo", self.repo], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"gh {' '.join(args[:2])} failed: {p.stderr.strip()[:300]}")
        return p.stdout

    def open_issues(self):
        out = self.run("issue", "list", "--state", "open", "--search", f'"{self.title.strip()}" in:title', "--json", "number,title", "--limit", "30")
        return sorted((i["number"] for i in json.loads(out or "[]") if i["title"].startswith(self.title)), reverse=True)

    def body_file(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        f.write(text)
        f.close()
        return f.name


def status_body(box, backend, verdict, sha, lines, now):
    """The standing status issue's body: one machine-readable `completed:` line (read by scripts/device_suite_heartbeat.py) and the runner's summary."""
    names = names_block(lines)
    return (f"completed: {now}\nbox: {box}\nbackend: {backend}\nverdict: {verdict}\nsha: {sha}\n\n"
            "Rewritten by every completed run of the device suite; a run that dies before a verdict leaves it as it was.\n\n```\n" + "\n".join(names) + "\n```\n")


def heartbeat(gh, box, backend, verdict, sha, lines, now):
    """Create or rewrite the box's standing status issue, red or green."""
    title = STATUS_PREFIX + box
    out = gh.run("issue", "list", "--state", "open", "--search", f'"{title}" in:title', "--json", "number,title", "--limit", "30")
    nums = sorted(i["number"] for i in json.loads(out or "[]") if i["title"] == title)
    body = gh.body_file(status_body(box, backend, verdict, sha, lines, now))
    if nums:
        gh.run("issue", "edit", str(nums[0]), "--body-file", body)
    else:
        gh.run("issue", "create", "--title", title, "--body-file", body)


def body_for(sha, lines, name="cuda-device-suite"):
    names = names_block(lines) or ["(the runner printed no summary)"]
    tail = lines[-TAIL_LINES:]
    return scrub(f"Revision `{sha}` of `origin/main`, run by the device job `{name}` with the device present "
                 f"(docs/measurements/{'metal-device-suite-2026-11-17' if name.startswith('metal') else 'cuda-device-suite-2026-10-18'}/README.md).\n\n```\n" + "\n".join(names) + "\n```\n\n"
                 f"Last {len(tail)} lines of the log:\n\n```\n" + "\n".join(tail) + "\n```\n")


def report(verdict, sha, lines, gh):
    open_now = gh.open_issues()
    if verdict == "RED":
        body = gh.body_file(body_for(sha, lines, gh.title.removesuffix(" red on ")))
        if open_now:
            gh.run("issue", "comment", str(open_now[0]), "--body-file", body)
            return f"commented on #{open_now[0]}"
        gh.run("issue", "create", "--title", gh.title + sha, "--body-file", body)
        return "opened an issue"
    done = []
    for n in open_now:
        gh.run("issue", "close", str(n), "--comment", f"{gh.title.removesuffix(' red on ')} is green on `{sha}`.")
        done.append(f"#{n}")
    return "closed " + ", ".join(done) if done else "green, nothing open"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict-log", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--repo", default="townsendmerino/goinfer")
    ap.add_argument("--gh", default="gh")
    ap.add_argument("--box", help="also rewrite the standing 'device-suite status: <box>' issue (the run-completed heartbeat)")
    ap.add_argument("--backend", default="cuda", help="cuda or metal: names the red issue and the runner's summary line")
    a = ap.parse_args(argv)
    lines = open(a.verdict_log, errors="replace").read().splitlines()
    verdict = parse_verdict(lines)
    if verdict is None:
        print("no VERDICT line in the log; no issue opened or closed")
        return 2
    try:
        gh = Gh(a.gh, a.repo, f"{a.backend}-device-suite red on ")
        print(f"{verdict}: {report(verdict, a.sha, lines, gh)}")
        if a.box:
            heartbeat(gh, a.box, a.backend, verdict, a.sha, lines, datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            print(f"heartbeat written for {a.box}")
    except RuntimeError as e:
        print(f"issue update failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
