#!/usr/bin/env python3
"""The device-suite heartbeat: a completed run rewrites "device-suite status: <box>", and scripts/device_suite_heartbeat.py files an issue when no completed run is
newer than 36 h. Fed a fake gh. The history replay is the failure that went unseen: nobara's last trace was 2026-11-04 and nothing said so for 13 nights.
Run: python3 -B scripts/test_device_suite_heartbeat.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import device_suite_heartbeat as hb  # noqa: E402

GREEN = ["cuda-device-suite: 91 passed, 0 failed, 0 device skips, 3 other skips, 0 failed package(s)", "VERDICT: GREEN"]

# A state-keeping fake gh: issues live in a JSON file; list filters on the title search, create/edit/comment/close mutate it and append to a call log.
FAKE = """#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
db = os.environ["FAKE_GH_DB"]
st = json.load(open(db))
with open(db + ".calls", "a") as f:
    f.write(json.dumps(a) + "\\n")
def val(flag): return a[a.index(flag) + 1]
if a[:2] == ["issue", "list"]:
    q = val("--search").split('"')[1]
    print(json.dumps([i for i in st if q in i["title"] and i["state"] == "open"]))
elif a[:2] == ["issue", "create"]:
    st.append({"number": 100 + len(st), "title": val("--title"), "body": open(val("--body-file")).read(), "state": "open"})
elif a[:2] == ["issue", "edit"]:
    for i in st:
        if i["number"] == int(a[2]): i["body"] = open(val("--body-file")).read()
elif a[:2] == ["issue", "close"]:
    for i in st:
        if i["number"] == int(a[2]): i["state"] = "closed"
elif a[:2] == ["issue", "comment"]:
    for i in st:
        if i["number"] == int(a[2]): i["body"] += "\\n" + open(val("--body-file")).read()
json.dump(st, open(db, "w"))
"""


class Fake:
    def __init__(self, issues=()):
        self.d = tempfile.mkdtemp()
        self.gh = os.path.join(self.d, "gh")
        self.db = os.path.join(self.d, "db.json")
        open(self.gh, "w").write(FAKE)
        os.chmod(self.gh, 0o755)
        json.dump(list(issues), open(self.db, "w"))
        os.environ["FAKE_GH_DB"] = self.db

    def issues(self):
        return json.load(open(self.db))

    def status_log(self, verdict_lines, box="nobara"):
        vl = os.path.join(self.d, "v.log")
        open(vl, "w").write("\n".join(verdict_lines) + "\n")
        return subprocess.run([sys.executable, os.path.join(HERE, "cuda_device_suite_issue.py"), "--verdict-log", vl, "--sha", "abc12345", "--box", box, "--gh", self.gh],
                              capture_output=True, text=True)

    def check(self, now, *extra):
        extra = extra if "--expect" in extra else ("--expect", "nobara", *extra)
        return subprocess.run([sys.executable, os.path.join(HERE, "device_suite_heartbeat.py"), "--now", now, "--gh", self.gh, *extra], capture_output=True, text=True)


def status(box, completed, n=7):
    return {"number": n, "title": hb.STATUS + box, "body": f"completed: {completed}\nbox: {box}\nverdict: GREEN\n", "state": "open"}


def stale_issues(f):
    return [i for i in f.issues() if i["title"].startswith(hb.STALE)]


class Completed(unittest.TestCase):
    def test_aGreenRunWithNoOtherTraceStillCreatesTheStatusIssue(self):
        f = Fake()
        p = f.status_log(GREEN)
        self.assertEqual(p.returncode, 0, p.stdout)
        [i] = f.issues()
        self.assertEqual(i["title"], "device-suite status: nobara")
        self.assertRegex(i["body"], r"(?m)^completed: \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertIn("verdict: GREEN", i["body"])

    def test_aSecondRunRewritesTheSameIssue(self):
        f = Fake([status("nobara", "2026-11-04T01:00:00Z")])
        self.assertEqual(f.status_log(GREEN).returncode, 0)
        [i] = f.issues()
        self.assertNotIn("2026-11-04", i["body"])

    def test_aLogWithNoVerdictWritesNoHeartbeat(self):
        f = Fake([status("nobara", "2026-11-04T01:00:00Z")])
        self.assertEqual(f.status_log(["FATAL: cannot check out origin/main"]).returncode, 2)
        self.assertIn("2026-11-04", f.issues()[0]["body"])


class Check(unittest.TestCase):
    def test_historyReplayThirteenSilentNightsIsStale(self):
        # The last trace of nobara's nightly was 2026-11-04; on 2026-11-17 nothing had said it stopped.
        f = Fake([status("nobara", "2026-11-04T01:30:00Z")])
        p = f.check("2026-11-17T18:00:00Z")
        self.assertEqual(p.returncode, 1, p.stdout)
        [s] = stale_issues(f)
        self.assertEqual(s["title"], "device-suite heartbeat stale: nobara")
        self.assertIn("2026-11-04", s["body"])

    def test_nightlyRunsKeepItFreshEachDayOfThatHistory(self):
        for day in range(5, 18):
            f = Fake([status("nobara", f"2026-11-{day:02d}T01:30:00Z")])
            p = f.check(f"2026-11-{day:02d}T18:00:00Z")
            self.assertEqual(p.returncode, 0, (day, p.stdout))
            self.assertEqual(stale_issues(f), [])

    def test_theLimitIs36Hours(self):
        self.assertEqual(Fake([status("nobara", "2026-11-16T06:00:00Z")]).check("2026-11-17T18:00:00Z").returncode, 0)
        self.assertEqual(Fake([status("nobara", "2026-11-16T05:59:00Z")]).check("2026-11-17T18:00:00Z").returncode, 1)

    def test_noStatusIssueAtAllIsStale(self):
        f = Fake()
        self.assertEqual(f.check("2026-11-17T18:00:00Z").returncode, 1)
        self.assertEqual(len(stale_issues(f)), 1)

    def test_aStaleBoxCommentsOnTheOpenIssueInsteadOfOpeningAnother(self):
        f = Fake([status("nobara", "2026-11-04T01:30:00Z")])
        f.check("2026-11-17T18:00:00Z")
        f.check("2026-11-18T18:00:00Z")
        self.assertEqual(len(stale_issues(f)), 1)

    def test_aFreshHeartbeatClosesTheStaleIssue(self):
        f = Fake([status("nobara", "2026-11-04T01:30:00Z")])
        f.check("2026-11-17T18:00:00Z")
        st = f.issues()
        for i in st:
            if i["title"] == hb.STATUS + "nobara":
                i["body"] = "completed: 2026-11-18T01:30:00Z\n"
        json.dump(st, open(f.db, "w"))
        self.assertEqual(f.check("2026-11-18T18:00:00Z").returncode, 0)
        self.assertEqual([i["state"] for i in stale_issues(f)], ["closed"])

    def test_eachExpectedBoxIsJudgedAlone(self):
        f = Fake([status("nobara", "2026-11-17T01:30:00Z"), status("mac", "2026-11-10T01:30:00Z", 8)])
        self.assertEqual(f.check("2026-11-17T18:00:00Z", "--expect", "nobara", "--expect", "mac").returncode, 1)
        self.assertEqual([i["title"] for i in stale_issues(f)], ["device-suite heartbeat stale: mac"])

    def test_aFailingGhIsReported(self):
        f = Fake()
        open(f.gh, "w").write("#!/bin/sh\necho boom >&2\nexit 1\n")
        self.assertEqual(f.check("2026-11-17T18:00:00Z").returncode, 3)


    def test_theDefaultExpectsBothBoxes(self):
        f = Fake([status("nobara", "2026-11-17T01:30:00Z")])
        p = subprocess.run([sys.executable, os.path.join(HERE, "device_suite_heartbeat.py"), "--now", "2026-11-17T18:00:00Z", "--gh", f.gh], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertEqual([i["title"] for i in stale_issues(f)], ["device-suite heartbeat stale: mac"])


class Plant(unittest.TestCase):
    def test_readingEveryBoxAsFreshFilesNothingForTheThirteenNightGap(self):
        saved = hb.judge
        hb.judge = lambda issues, expect, now, max_age: {b: (True, "plant") for b in expect}
        try:
            f = Fake([status("nobara", "2026-11-04T01:30:00Z")])
            rc = hb.main(["--now", "2026-11-17T18:00:00Z", "--gh", f.gh, "--expect", "nobara"])
            self.assertEqual(rc, 0)
            self.assertEqual(stale_issues(f), [], "the plant must file nothing, or the replay test cannot see the defect it plants")
        finally:
            hb.judge = saved


if __name__ == "__main__":
    unittest.main()
