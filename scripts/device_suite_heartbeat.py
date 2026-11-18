#!/usr/bin/env python3
"""Read the device suites' run-completed heartbeats and file an issue when one is stale.

Each completed run of a device suite rewrites a standing issue "device-suite status: <box>" whose body begins `completed: <UTC time>`
(scripts/cuda_device_suite_issue.py --box). A green run leaves no other trace, so a suite that stopped running looks the same as one that is green.
This check reads those issues (the scheduled workflow .github/workflows/device-heartbeat.yml runs it daily; a review loop may run it too) and, for each
expected box whose newest completed run is older than --max-age-h (36 by default), or that has no status issue, opens "device-suite heartbeat stale:
<box>" or comments on the open one; a fresh box closes its open stale issue. Stale is "unknown", which is red before a tag.

  python3 scripts/device_suite_heartbeat.py [--expect nobara --expect mac] [--max-age-h 36] [--repo townsendmerino/goinfer] [--gh gh] [--now ISO]

Exit 0 when every expected box is fresh, 1 when any is stale or missing (the issue is filed first), 3 when a gh call failed.
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile

STATUS = "device-suite status: "
STALE = "device-suite heartbeat stale: "
DEFAULT_EXPECT = ["nobara", "mac"]


def parse_time(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)


def completed_at(body):
    m = re.search(r"^completed: (\S+)\s*$", body or "", re.M)
    try:
        return parse_time(m.group(1)) if m else None
    except ValueError:
        return None


def judge(status_issues, expect, now, max_age_h):
    """status_issues: [{title, body}]. Returns {box: (fresh, text)}."""
    newest = {}
    for i in status_issues:
        if not i["title"].startswith(STATUS):
            continue
        t = completed_at(i.get("body"))
        box = i["title"][len(STATUS):]
        if t and (box not in newest or t > newest[box]):
            newest[box] = t
    res = {}
    for box in expect:
        t = newest.get(box)
        if t is None:
            res[box] = (False, f"no completed run on record for {box}: no `{STATUS}{box}` issue with a `completed:` line")
            continue
        age = (now - t).total_seconds() / 3600
        res[box] = (age <= max_age_h, f"{box}: last completed run {t:%Y-%m-%d %H:%M} UTC, {age:.1f} h ago (limit {max_age_h:g} h)")
    return res


class Gh:
    def __init__(self, gh, repo):
        self.gh, self.repo = gh, repo

    def run(self, *args):
        p = subprocess.run([self.gh, *args, "--repo", self.repo], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"gh {' '.join(args[:2])} failed: {p.stderr.strip()[:300]}")
        return p.stdout

    def issues(self, prefix):
        out = self.run("issue", "list", "--state", "open", "--search", f'"{prefix.strip()}" in:title', "--json", "number,title,body", "--limit", "50")
        return [i for i in json.loads(out or "[]") if i["title"].startswith(prefix)]

    def body_file(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        f.write(text)
        f.close()
        return f.name


def act(gh, res):
    open_stale = {i["title"][len(STALE):]: i["number"] for i in sorted(gh.issues(STALE), key=lambda i: i["number"])}
    for box, (fresh, text) in res.items():
        n = open_stale.get(box)
        if fresh and n:
            gh.run("issue", "close", str(n), "--comment", f"Heartbeat is fresh again. {text}")
        elif not fresh:
            body = gh.body_file(f"{text}\n\nThe device suite for `{box}` has not recorded a completed run inside its window, so its state is unknown: "
                                "treat it as red before a tag. Check the box's timer and queue (docs/measurements/cuda-device-suite-2026-10-18/README.md).\n")
            if n:
                gh.run("issue", "comment", str(n), "--body-file", body)
            else:
                gh.run("issue", "create", "--title", STALE + box, "--body-file", body)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expect", action="append", help="a box that must have a fresh heartbeat (repeatable; default nobara and mac)")
    ap.add_argument("--max-age-h", type=float, default=36)
    ap.add_argument("--repo", default="townsendmerino/goinfer")
    ap.add_argument("--gh", default="gh")
    ap.add_argument("--now", help="UTC time, %%Y-%%m-%%dT%%H:%%M:%%SZ (tests)")
    a = ap.parse_args(argv)
    now = parse_time(a.now) if a.now else datetime.datetime.now(datetime.timezone.utc)
    gh = Gh(a.gh, a.repo)
    try:
        res = judge(gh.issues(STATUS), a.expect or DEFAULT_EXPECT, now, a.max_age_h)
        for _, text in res.values():
            print(text)
        act(gh, res)
    except RuntimeError as e:
        print(f"heartbeat check failed: {e}")
        return 3
    return 0 if all(f for f, _ in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
