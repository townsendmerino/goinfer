#!/usr/bin/env python3
"""transcript_miner.py — TE0's complete census: every command a Claude Code session ran on this machine, with its
duration, from the session transcripts (docs/tasks/task-test-efficiency-2026-09.md, TE0 "Complete").

WHY. census.py sees only what was archived. A daytime `go test` that printed to a terminal, a poll loop, or a smoke run
never reaches a log. Every one of them is in the transcripts: each Bash tool call is a `tool_use` entry (timestamp,
command, run_in_background) and a `tool_result` entry with the same id (timestamp). A background command, or one
that overran its timeout, finishes later in a `<task-notification>` naming its task id.

Read-only. It runs nothing and times nothing.

  usage: transcript_miner.py [--since YYYY-MM-DD] [--projects GLOB ...] [--day HH-HH] [--top N]
         defaults: ~/.claude/projects/*tmcode-goinfer* and *tmcode-aikit* (subagent transcripts included),
                   the last 30 days, day = 07-22 local

DURATION of a call = its result's timestamp − its call's timestamp; for a backgrounded command, the matching
task-notification's timestamp − the call's. LIMITS, so nobody reads more into the tables than is there:
  - A call launched DETACHED (launchctl submit, setsid/nohup … &) returns at once. Its real run is in its own log, which
    census.py reads (START/END spans). Here it is counted as "detached launch" with no duration.
  - Parallel calls overlap, so class sums can exceed wall-clock. The union line merges each session's intervals.
  - Wait/poll calls (until … sleep, gh run watch loops, Monitor waits) are wall time the owner sat through, not
    machine work. They get their own class so they can be read either way.
  - A foreground call whose result arrives more than 610 s later was BLOCKED (a permission prompt, an unattended or
    suspended session), because the Bash tool caps a foreground call at 600 s. Those calls are listed and excluded.
  - Work launched on nobara over ssh from this Mac is attributed to nobara; nobara's own sessions are in its own
    transcripts, which a session there has to mine.
"""
import argparse
import collections
import datetime as dt
import glob
import json
import os
import platform
import re
import sys

FG_MAX_S = 610  # the Bash tool's 600 s cap, plus slack
BG_ID = re.compile(r"(?:running in background with ID|moved to the background \(ID): ?([a-z0-9]+)")
NOTE = re.compile(r"<task-id>([a-z0-9]+)</task-id>.*?<status>(completed|failed|killed|stopped)</status>", re.S)

# First match wins. Ordered from most specific to least.
CLASSES = [
    ("detached launch", re.compile(r"launchctl submit|\bsetsid\b|\bnohup\b.*&\s*$|night\.py start")),
    ("wait/poll", re.compile(r"^\s*(?:L=\S+;\s*)?(?:until |while |for i in \$\(seq)|^\s*sleep \d|gh run watch")),
    ("served gate", re.compile(r"bench_peer\w*\.py|run-[\w.-]+\.sh")),
    ("parity/refresh", re.compile(r"refresh_parity_hashes|cmd/gate\s+parity|gate parity")),
    ("gate", re.compile(r"go run \S*cmd/gate\b|go run -C tools \./\w+")),
    ("in-test measurement", re.compile(r"go test[^|;&]*(?:-bench\b|GOINFER_HEAVY_TESTS=1|realckpt|Gate|Reference|endToEnd|"
                                       r"Measured|Mechanism|_bench)|GOINFER_HEAVY_TESTS=1[^|;&]*go test")),
    ("test suite", re.compile(r"\bgo test\b")),
    ("evaluation", re.compile(r"\bdecide\b|decisions-calibrate|run-d6a|analyze\.py")),
    ("logit dump / identity", re.compile(r"f16xbuild|xb-(?:cpu|metal|webgpu)|\bcmp\b.*dump")),
    ("build/lint", re.compile(r"\bgo (?:build|vet|install)\b|gofmt|staticcheck|golangci-lint|py_compile|citation_lint")),
    ("git/gh", re.compile(r"^\s*(?:cd \S+ && )?(?:git|gh) ")),
]


LOOP_SLEEP = re.compile(r"\b(?:until|while|for)\b[\s\S]{0,600}?\bsleep\b")  # loops span lines in a heredoc'd script
DOES_WORK = re.compile(r"python3 \S*bench_peer|\bgo test\b|bash \S*run-[\w.-]+\.sh|refresh_parity_hashes|go run \S*cmd/gate")


def classify(cmd):
    # A loop that sleeps is a wait, unless it also runs the work itself (a run wrapper's idle loop before its harness).
    if LOOP_SLEEP.search(cmd) and not DOES_WORK.search(cmd):
        return "wait/poll"
    for name, rx in CLASSES:
        if rx.search(cmd):
            return name
    return "other"


def machine(cmd):
    return "nobara (via ssh)" if re.match(r"\s*(?:cd \S+ && )?(?:timeout \d+ )?ssh\b", cmd) else "local"


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def mine(files, since):
    calls = {}        # tool_use id -> record
    bg = {}           # background task id -> tool_use id
    for f in files:
        sess = os.path.basename(os.path.dirname(os.path.dirname(f))) if "/subagents/" in f else os.path.basename(f)[:-6]
        agent = os.path.basename(f)[:-6] if "/subagents/" in f else ""
        with open(f, errors="replace") as fh:
            for line in fh:
                if '"tool_use"' not in line and '"tool_result"' not in line and "task-notification" not in line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                t = e.get("timestamp")
                if not t:
                    continue
                when = ts(t)
                content = (e.get("message") or {}).get("content")
                if not isinstance(content, list):
                    if isinstance(content, str) and "task-notification" in content:
                        for tid, status in NOTE.findall(content):
                            if tid in bg and "end" not in calls[bg[tid]]:
                                calls[bg[tid]].update(end=when, status=status)
                    continue
                for c in content:
                    ctype = c.get("type")
                    if ctype == "tool_use" and c.get("name") == "Bash":
                        inp = c.get("input") or {}
                        if when < since:
                            continue
                        calls[c["id"]] = {"start": when, "cmd": inp.get("command", ""), "bg": bool(inp.get("run_in_background")),
                                          "session": sess, "agent": agent, "file": f}
                    elif ctype == "tool_result":
                        rec = calls.get(c.get("tool_use_id"))
                        if rec is None or "result" in rec:
                            continue
                        rec["result"] = when
                        text = c.get("content")
                        if isinstance(text, list):
                            text = " ".join(x.get("text", "") for x in text if isinstance(x, dict))
                        m = BG_ID.search(text or "")
                        if m:
                            bg[m.group(1)] = c["tool_use_id"]
                            rec["bgid"] = m.group(1)
                        else:
                            rec["end"] = when
                    elif ctype == "text" and "task-notification" in (c.get("text") or ""):
                        for tid, status in NOTE.findall(c["text"]):
                            if tid in bg and "end" not in calls[bg[tid]]:
                                calls[bg[tid]].update(end=when, status=status)
    out = []
    for rec in calls.values():
        rec["class"] = classify(rec["cmd"])
        rec["machine"] = machine(rec["cmd"])
        end = rec.get("end")
        rec["secs"] = (end - rec["start"]).total_seconds() if end else None
        # A foreground Bash call cannot run past the tool's 600 s maximum timeout (it is moved to the background, and
        # that is caught above). A longer gap is the call BLOCKED (a permission prompt, an unattended or suspended
        # session), not run time: it is kept, marked, and left out of every sum.
        if rec["secs"] is not None and not rec["bg"] and "bgid" not in rec and rec["secs"] > FG_MAX_S:
            rec["stalled"], rec["secs"] = rec["secs"], None
        # A background task that ended "stopped" ran until someone stopped it (TaskStop, or the session ending): a
        # server left up for days, a watch nobody needed. Its span is not the work's duration, so it is excluded too.
        if rec["secs"] is not None and rec.get("status") == "stopped":
            rec["stalled"], rec["secs"] = rec["secs"], None
        out.append(rec)
    return out


def union_seconds(intervals):
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += (cur_e - cur_s).total_seconds()
    return tot


def hm(secs):
    return f"{secs / 3600:6.1f} h" if secs >= 3600 else f"{secs / 60:6.1f} m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=(dt.date.today() - dt.timedelta(days=30)).isoformat())
    ap.add_argument("--projects", nargs="*", default=[os.path.expanduser("~/.claude/projects/*tmcode-goinfer*"),
                                                       os.path.expanduser("~/.claude/projects/*tmcode-aikit*")])
    ap.add_argument("--day", default="07-22", help="local hours counted as day, HH-HH")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    since = dt.datetime.fromisoformat(a.since).astimezone()
    d0, d1 = (int(x) for x in a.day.split("-"))
    files = sorted({f for p in a.projects for d in glob.glob(p) for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)})
    recs = mine(files, since)
    timed = [r for r in recs if r["secs"] is not None]
    local = lambda t: t.astimezone()
    isday = lambda t: d0 <= local(t).hour < d1

    print(f"# transcript census — {platform.node()} — {dt.datetime.now().astimezone():%Y-%m-%d %H:%M %Z}")
    print(f"{len(files)} transcript files; {len(recs)} Bash calls since {a.since}; {len(timed)} with a duration; "
          f"{len(recs) - len(timed)} without (detached launches, or a result never recorded). Day = {d0:02d}-{d1:02d} local.\n")

    print("## 1. by class (sum of call durations; parallel calls overlap)\n")
    print(f"| class | calls | total | day | night | median | longest |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    by = collections.defaultdict(list)
    for r in recs:
        by[r["class"]].append(r)
    rows = []
    for k, rs in by.items():
        ss = sorted(r["secs"] for r in rs if r["secs"] is not None)
        tot = sum(ss)
        day = sum(r["secs"] for r in rs if r["secs"] is not None and isday(r["start"]))
        rows.append((tot, k, len(rs), day, tot - day, ss[len(ss) // 2] if ss else 0, ss[-1] if ss else 0))
    for tot, k, n, day, night, med, mx in sorted(rows, reverse=True):
        print(f"| {k} | {n} | {hm(tot)} | {hm(day)} | {hm(night)} | {hm(med)} | {hm(mx)} |")
    allsum = sum(r["secs"] for r in timed)
    per_sess = collections.defaultdict(list)
    for r in timed:
        per_sess[r["session"] + r["agent"]].append((r["start"], r["start"] + dt.timedelta(seconds=r["secs"])))
    uni = sum(union_seconds(v) for v in per_sess.values())
    print(f"\nAll classes: {hm(allsum)} of call time; {hm(uni)} after merging each session's overlapping calls.")
    st = [r for r in recs if r.get("stalled")]
    print(f"Excluded: {len(st)} call(s), {hm(sum(r['stalled'] for r in st))} in total: foreground calls whose result came more "
          f"than {FG_MAX_S} s later (blocked, since a foreground call cannot run that long), and background tasks that "
          f"ended 'stopped' (they ran until someone stopped them, so the span is not the work).")
    for mch in ("local", "nobara (via ssh)"):
        ms = sum(r["secs"] for r in timed if r["machine"] == mch and r["class"] != "wait/poll")
        print(f"  work (excluding wait/poll) run {mch}: {hm(ms)}")
    print()

    print("## 2. by day (sum of call durations, excluding wait/poll)\n")
    print("| date | calls | work | of which served / in-test / suite / parity | wait/poll |")
    print("|---|---:|---:|---|---:|")
    days = collections.defaultdict(lambda: collections.Counter())
    for r in timed:
        days[local(r["start"]).date()][r["class"]] += r["secs"]
        days[local(r["start"]).date()]["_n"] += 1
    for d in sorted(days):
        c = days[d]
        work = sum(v for k, v in c.items() if not k.startswith("_") and k != "wait/poll")
        print(f"| {d} | {c['_n']} | {hm(work)} | {hm(c['served gate'])} / {hm(c['in-test measurement'])} / "
              f"{hm(c['test suite'])} / {hm(c['parity/refresh'])} | {hm(c['wait/poll'])} |")

    print(f"\n## 3. the {a.top} longest calls\n")
    print("| started (local) | dur | class | bg | command |")
    print("|---|---:|---|---|---|")
    for r in sorted(timed, key=lambda r: -r["secs"])[:a.top]:
        cmd = " ".join(r["cmd"].split())[:110].replace("|", "\\|")
        print(f"| {local(r['start']):%m-%d %H:%M} | {hm(r['secs'])} | {r['class']} | {'y' if r['bg'] else ''} | `{cmd}` |")

    print("\n## 4. re-runs: the same test/bench command run 3+ times in one day\n")
    print("| date | runs | total | class | command |")
    print("|---|---:|---:|---|---|")
    norm = lambda c: re.sub(r"\s+", " ", re.sub(r"/private/tmp/\S+|/tmp/\S+|\b\d{2}:\d{2}:\d{2}\b", "<tmp>", c)).strip()[:160]
    reruns = collections.defaultdict(list)
    for r in timed:
        if r["class"] in ("served gate", "in-test measurement", "test suite", "parity/refresh", "gate", "logit dump / identity"):
            reruns[(local(r["start"]).date(), norm(r["cmd"]))].append(r)
    for (d, c), rs in sorted(reruns.items(), key=lambda kv: -sum(r["secs"] for r in kv[1])):
        if len(rs) >= 3:
            print(f"| {d} | {len(rs)} | {hm(sum(r['secs'] for r in rs))} | {rs[0]['class']} | `{c[:100].replace('|', chr(92) + '|')}` |")

    det = [r for r in recs if r["class"] == "detached launch"]
    print(f"\n## 5. detached launches (duration in their own logs, see census.py): {len(det)}\n")
    for r in det[:15]:
        print(f"- {local(r['start']):%m-%d %H:%M} `{' '.join(r['cmd'].split())[:140]}`")


if __name__ == "__main__":
    sys.exit(main())
