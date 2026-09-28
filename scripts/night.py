#!/usr/bin/env python3
"""night.py — the overnight queue for long runs. Quick by day, long by night (owner rule, 2026-09-28).

WHY. The owner works on the MacBook during the day and wants quick turnarounds. Before this, sessions
launched every timed gate the moment it was ready. On 2026-09-27/28 the L1 campaign ran six 26-46 minute
timed sweeps across the two machines, three of them on the Mac between 07:15 and 09:05 while he was at the
keyboard, and the Mac's 41-minute served pass spent 29 of those minutes in bench_peer.py's per-cell idle
gate waiting out load that his own editor and desktop were making. A run like that is slow AND a weaker measurement (the load cap had to be
raised to 2.5 as a disclosed deviation). At night the box is quiet and nobody is waiting.

So sessions QUEUE long work here during the day, and the owner starts the queue at bedtime. The queue
runs one job at a time, detached (it survives the terminal and every Claude session), keeps the Mac awake
while it works, stops starting jobs that would run past the morning deadline, and leaves a SUMMARY.md to
pick up from. See root CLAUDE.md, "Run budget: quick by day, long by night".

usage:
  night.py add NAME --est MIN [--timeout MIN] [--priority N] [--by WHO] [--doc PATH] -- CMD [ARGS...]
                                     (refuses a command already queued from the same directory)
  night.py list [--from HH:MM]       the queue in run order, with estimates and projected finish
  night.py start [--deadline HH:MM]  bedtime: detach and run the queue (default deadline 06:30)
  night.py status                    what is running now and what finished tonight
  night.py stop [--now]              finish the current job, then stop  (--now: kill it too)
  night.py morning                   print the latest night's SUMMARY.md
  night.py drop NAME                 remove a queued job

A job is a SELF-CONTAINED command, normally a docs/measurements/<campaign>/run-*.sh: pinned revs or
binaries, its own durable log/record paths, nothing interactive, no Claude session needed. It runs with
the environment of whoever ran `start`, in the directory it was queued from. Its stdout/stderr go to
<night dir>/runs/<date>/<NAME>.log in addition to whatever logs the job writes itself.

Everything lives under $NIGHT_DIR (default ~/goinfer-logs/night): queue/, runs/<date>/, runner.pid.
The queue is per machine — the Mac and nobara each have their own.
"""
import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import timing_lock  # TE9: one timed run per box

ROOT = Path(os.environ.get("NIGHT_DIR", str(Path.home() / "goinfer-logs" / "night"))).expanduser()
QUEUE = ROOT / "queue"
RUNS = ROOT / "runs"
RUNNING = ROOT / "running"        # the job file while it runs; back to queue/ if the runner died mid-job
PIDFILE = ROOT / "runner.pid"
STATE = ROOT / "state.json"      # the job running right now
STOPFLAG = ROOT / "STOP"         # finish the current job, then stop
DEFAULT_DEADLINE = os.environ.get("NIGHT_DEADLINE", "06:30")
MAX_EST_MIN = 360                # "never a 6-hour run" — split it or fix the slow path first
SPLIT_WARN_MIN = 180
SETTLE_LOADAVG = float(os.environ.get("NIGHT_SETTLE_LOADAVG", "1.0"))
SETTLE_MAX_S = int(os.environ.get("NIGHT_SETTLE_MAX_S", "300"))
POLL_S = 10
LOCK_WAIT_MIN = float(os.environ.get("NIGHT_LOCK_WAIT_MIN", "30"))  # TE9: minutes a job waits for another timed run's lock
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


# ---------------------------------------------------------------- helpers

def now():
    return dt.datetime.now().astimezone()


def hm(t):
    return t.strftime("%H:%M")


def fmt_min(m):
    m = int(round(m))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def die(msg, code=1):
    print(f"night: {msg}", file=sys.stderr)
    sys.exit(code)


def git(cwd, *args):
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")   # never contend with a live session's index.lock
        out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=20, env=env)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(p)


def queued():
    """Queued jobs in run order: priority (lower first), then the order they were added."""
    jobs = []
    for f in sorted(QUEUE.glob("*.json")) if QUEUE.exists() else []:
        j = read_json(f)
        if j:
            j["_file"] = str(f)
            jobs.append(j)
    jobs.sort(key=lambda j: (j.get("priority", 50), j.get("seq", 0)))
    return jobs


def running():
    """The job file of whatever is running right now (empty by day, normally)."""
    out = []
    for f in sorted(RUNNING.glob("*.json")) if RUNNING.exists() else []:
        j = read_json(f)
        if j:
            out.append(j)
    return out


def runner_pid():
    try:
        pid = int(PIDFILE.read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
        return pid
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid


def parse_deadline(s, start):
    try:
        h, m = (int(x) for x in s.split(":"))
    except Exception:
        die(f"--deadline wants HH:MM, got {s!r}")
    d = start.replace(hour=h, minute=m, second=0, microsecond=0)
    if d <= start:
        d += dt.timedelta(days=1)
    return d


def latest_run_dir():
    if not RUNS.exists():
        return None
    dirs = sorted(d for d in RUNS.iterdir() if d.is_dir())
    return dirs[-1] if dirs else None


# ---------------------------------------------------------------- add / list / drop

def cmd_add(a):
    if not NAME_RE.match(a.name):
        die("NAME: letters, digits, . _ - (up to 80), starting with a letter or digit")
    cmd = a.cmd
    if not cmd:
        die("nothing to run — put the command after `--`")
    if a.est <= 0:
        die("--est is the estimated runtime in minutes, and it is required: the runner plans the night with it")
    if a.est > MAX_EST_MIN and not a.force:
        die(f"--est {a.est} min is over {MAX_EST_MIN}. Owner rule: never a 6-hour run. Split it, shrink the "
            f"sample, or fix the slow path first (D6a did both). --force if the owner asked for it.")
    QUEUE.mkdir(parents=True, exist_ok=True)
    if any(j["name"] == a.name for j in queued()):
        die(f"a job named {a.name!r} is already queued (night.py drop {a.name} first)")
    cwd = Path(a.cwd or os.getcwd()).resolve()
    # ONE COPY OF EACH RUN. The same command from the same directory is the same run tonight, whatever rev each
    # session was on when it queued it: a tree-relative command runs on tonight's tree, and a run-*.sh runs
    # whatever that script says tonight. Two sessions queueing the same gate is the case this exists for.
    if not a.allow_duplicate:
        for j in queued() + running():
            if j["cmd"] == cmd and j["cwd"] == str(cwd):
                die(f"the same command from the same directory is already queued as {j['name']!r}"
                    f"{' by ' + j['by'] if j.get('by') else ''} (est {fmt_min(j['est_min'])}). It runs on tonight's "
                    f"tree, so a second copy would repeat it — name that job in your report instead. A repeat that "
                    f"is part of a pre-registered design goes in with --allow-duplicate.")
    seq = int(time.time() * 1000)
    job = {
        "name": a.name,
        "cmd": cmd,
        "cwd": str(cwd),
        "est_min": a.est,
        "timeout_min": a.timeout or max(2 * a.est, a.est + 20),
        "priority": a.priority,
        "by": a.by or "",
        "doc": a.doc or "",
        "note": a.note or "",
        "rev": git(cwd, "rev-parse", "--short", "HEAD"),
        "dirty": bool(git(cwd, "status", "--porcelain", "--untracked-files=no")),
        "host": platform.node(),
        "added": now().isoformat(timespec="seconds"),
        "seq": seq,
    }
    write_json(QUEUE / f"{seq}-{a.name}.json", job)
    if a.est > SPLIT_WARN_MIN:
        print(f"night: note — {a.est} min is over {SPLIT_WARN_MIN}; prefer jobs of 3 h or less so a failure "
              f"costs one job, not the night.", file=sys.stderr)
    jobs = queued()
    total = sum(j["est_min"] for j in jobs)
    print(f"queued {a.name!r} (est {fmt_min(a.est)}, timeout {fmt_min(job['timeout_min'])}, rev {job['rev'] or '?'}"
          f"{' +dirty' if job['dirty'] else ''}). Tonight's queue: {len(jobs)} job(s), {fmt_min(total)}.")


def cmdline(j, width=110):
    """The command as a shell line, shown by `list` so a session can see whether a queued job already covers its run."""
    c = shlex.join(j["cmd"])
    if j["cwd"] != os.getcwd():
        c = f"(in {j['cwd']}) {c}"
    return c if len(c) <= width else c[:width - 1] + "…"


def plan(jobs, start, deadline):
    """Greedy in run order: a job starts only if its estimate ends by the deadline."""
    t, rows, left = start, [], []
    for j in jobs:
        end = t + dt.timedelta(minutes=j["est_min"])
        if end <= deadline:
            rows.append((j, t, end))
            t = end
        else:
            left.append(j)
    return rows, left


def cmd_list(a):
    jobs = queued()
    if not jobs:
        print("the night queue is empty")
        return
    start = now()
    frm = getattr(a, "frm", None)
    if frm:
        start = parse_deadline(frm, start - dt.timedelta(minutes=1))
    deadline = parse_deadline(a.deadline, start)
    rows, left = plan(jobs, start, deadline)
    total = sum(j["est_min"] for j in jobs)
    print(f"{len(jobs)} queued job(s), {fmt_min(total)} estimated. If started {'at' if frm else 'now,'} "
          f"{hm(start)}, deadline {hm(deadline)}:")
    print(f"  {'pri':>3}  {'job':<34} {'est':>10}  {'start':>5}-{'end':<5}  by / doc")
    for j, s, e in rows:
        who = " / ".join(x for x in (j.get("by"), j.get("doc")) if x)
        print(f"  {j['priority']:>3}  {j['name']:<34} {fmt_min(j['est_min']):>10}  {hm(s)}-{hm(e):<5}  {who}")
        print(f"       $ {cmdline(j)}")
    for j in left:
        print(f"  {j['priority']:>3}  {j['name']:<34} {fmt_min(j['est_min']):>10}  would not fit before {hm(deadline)}"
              f" — stays queued")
        print(f"       $ {cmdline(j)}")
    latest = deadline - dt.timedelta(minutes=total)
    if latest > start:
        print(f"Everything fits if started by {hm(latest)}.")
    pid = runner_pid()
    if pid:
        print(f"(a runner is already going, pid {pid} — `night.py status`)")


def cmd_drop(a):
    hits = [j for j in queued() if j["name"] == a.name]
    if not hits:
        die(f"no queued job named {a.name!r}")
    for j in hits:
        Path(j["_file"]).unlink()
    print(f"dropped {a.name!r}")


# ---------------------------------------------------------------- start / run

def cmd_start(a):
    pid = runner_pid()
    if pid:
        die(f"a runner is already going (pid {pid}). `night.py status`, or `night.py stop`.")
    jobs = queued()
    if not jobs:
        print("the night queue is empty — nothing to start")
        return
    start = now()
    deadline = parse_deadline(a.deadline, start)
    rows, left = plan(jobs, start, deadline)
    cmd_list(a)
    if not rows:
        die(f"nothing fits before {hm(deadline)}. Pass a later --deadline, or split the jobs.")
    if a.dry_run:
        print("(dry run — not started)")
        return
    if sys.platform == "darwin":
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        if "Battery Power" in batt:
            print("WARNING: this Mac is on battery. Plug it in: keeping it awake overnight only holds on AC power.")
        print("Mac: leave it plugged in with the lid open (the screen can go dark). Closing the lid sleeps it.")
    STOPFLAG.unlink(missing_ok=True)   # a stale STOP from last night must not stop tonight
    day = start.strftime("%Y-%m-%d")
    rundir = RUNS / day
    rundir.mkdir(parents=True, exist_ok=True)
    log = open(rundir / "runner.log", "a")
    here = Path(__file__).resolve()
    child = subprocess.Popen(
        [sys.executable, str(here), "_run", "--deadline", deadline.isoformat(), "--rundir", str(rundir)],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True, cwd=str(ROOT))
    time.sleep(1.5)
    if child.poll() is not None:
        if child.returncode == 0:
            print(f"the queue already ran to completion — {rundir / 'SUMMARY.md'}")
            return
        die(f"the runner exited at once (rc {child.returncode}) — see {rundir / 'runner.log'}")
    print(f"started: runner pid {child.pid}, detached (own session — closing this terminal does not stop it).")
    print(f"logs and SUMMARY.md: {rundir}")
    print("In the morning: `python3 scripts/night.py morning`.")


class Runner:
    def __init__(self, deadline, rundir):
        self.deadline = deadline
        self.rundir = Path(rundir)
        self.jobsdir = self.rundir / "jobs"
        self.jobsdir.mkdir(parents=True, exist_ok=True)
        self.started = now()
        self.proc = None
        self.stop_now = False
        self.awake = None
        self.results = [read_json(f) for f in sorted(self.jobsdir.glob("*.json"))]
        self.results = [r for r in self.results if r]
        self.why_left = {}

    def log(self, msg):
        print(f"[{now().strftime('%F %T %Z')}] {msg}", flush=True)

    def keep_awake(self):
        """A side process that holds a no-sleep assertion for as long as this runner lives."""
        me = str(os.getpid())
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            cmd = ["caffeinate", "-i", "-s", "-w", me]
        elif shutil.which("systemd-inhibit"):
            cmd = ["systemd-inhibit", "--what=sleep:idle", "--who=goinfer-night", "--why=overnight queue",
                   "--mode=block", "/bin/sh", "-c", f"while kill -0 {me} 2>/dev/null; do sleep 30; done"]
        else:
            self.log("no caffeinate / systemd-inhibit on PATH — nothing is keeping this machine awake")
            return
        try:
            self.awake = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
            time.sleep(1)
            if self.awake.poll() is not None:
                self.log(f"WARNING: {cmd[0]} exited at once (rc {self.awake.returncode}) — the machine may sleep")
                self.awake = None
            else:
                self.log(f"keeping the machine awake: {' '.join(cmd)}")
        except Exception as e:
            self.log(f"WARNING: could not start {cmd[0]}: {e}")

    def on_term(self, *_):
        self.stop_now = True

    def settle(self):
        """Give the previous job's load a few minutes to drain; the job's own idle gates still decide."""
        waited = 0
        while waited < SETTLE_MAX_S:
            la = os.getloadavg()[0]
            if la <= SETTLE_LOADAVG:
                return
            if waited == 0:
                self.log(f"settling: loadavg {la:.2f} > {SETTLE_LOADAVG} (up to {SETTLE_MAX_S}s)")
            time.sleep(15)
            waited += 15
        self.log(f"settle gave up after {SETTLE_MAX_S}s at loadavg {os.getloadavg()[0]:.2f}; starting anyway")

    def kill_job(self, why):
        if not self.proc or self.proc.poll() is not None:
            return
        self.log(f"stopping job ({why}): SIGTERM to its process group")
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(30):
            if self.proc.poll() is not None:
                return
            time.sleep(1)
        self.log("still alive after 30 s: SIGKILL")
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def run_job(self, j):
        name = j["name"]
        logp = self.rundir / f"{name}.log"
        cwd = j["cwd"]
        RUNNING.mkdir(parents=True, exist_ok=True)
        moved = RUNNING / Path(j["_file"]).name
        Path(j["_file"]).replace(moved)
        j["_file"] = str(moved)
        rev_now = git(cwd, "rev-parse", "--short", "HEAD")
        t0 = now()
        write_json(STATE, {"name": name, "started": t0.isoformat(timespec="seconds"), "est_min": j["est_min"],
                           "timeout_min": j["timeout_min"], "log": str(logp), "pid": None})
        with open(logp, "a") as f:
            f.write(f"=== night job {name}\n=== cmd      {shlex.join(j['cmd'])}\n=== cwd      {cwd}\n"
                    f"=== rev      queued at {j.get('rev') or '?'}{' +dirty' if j.get('dirty') else ''}, "
                    f"now {rev_now or '?'}\n=== host     {platform.node()}\n"
                    f"=== est      {j['est_min']} min, timeout {j['timeout_min']} min\n")
            for k in ("by", "doc", "note"):
                if j.get(k):
                    f.write(f"=== {k:<8} {j[k]}\n")
            f.write(f"=== START    {t0.strftime('%F %T %Z')}  loadavg {os.getloadavg()[0]:.2f}\n")
            f.flush()
            # TE9: the job holds the timing lock, and its children (a run-*.sh, bench_peer.py) inherit it through the
            # environment. A timed run still going from the day, or from another checkout, is waited for up to
            # LOCK_WAIT_MIN; then this job is skipped, naming the holder, rather than measured on top of it.
            try:
                lock = timing_lock.acquire(label=f"night.py job {name}", wait_s=LOCK_WAIT_MIN * 60,
                                           cmd=shlex.join(j["cmd"]))
            except timing_lock.Held as e:
                f.write(f"=== could not start: {e}\n")
                self.proc = None
                return self.finish(j, t0, "skipped: timing lock held", logp)
            try:
                return self._run_locked(j, name, cwd, t0, logp, f)
            finally:
                lock.release()

    def _run_locked(self, j, name, cwd, t0, logp, f):
        """The job itself: run_job calls this with the log open and the timing lock held."""
        cmd = j["cmd"] if len(j["cmd"]) > 1 else ["/bin/bash", "-c", j["cmd"][0]]
        env = dict(os.environ, NIGHT_JOB=name, NIGHT_LOG=str(logp), NIGHT_RUNDIR=str(self.rundir))
        try:
            if not Path(cwd).is_dir():
                raise FileNotFoundError(f"the directory it was queued from is gone: {cwd}")
            self.proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=f,
                                         stderr=subprocess.STDOUT, start_new_session=True, env=env)
        except Exception as e:
            f.write(f"=== could not start: {e}\n")
            self.proc = None
            return self.finish(j, t0, "failed to start", logp)
        st = read_json(STATE) or {}
        st["pid"] = self.proc.pid
        write_json(STATE, st)
        limit = t0 + dt.timedelta(minutes=j["timeout_min"])
        outcome = None
        while True:
            try:
                self.proc.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            if self.stop_now:
                self.kill_job("night.py stop --now")
                outcome = "stopped"
            elif now() > limit:
                self.kill_job(f"timeout {j['timeout_min']} min")
                outcome = "timeout"
        rc = self.proc.returncode
        f.write(f"\n=== END      {now().strftime('%F %T %Z')}  rc={rc}\n")
        f.flush()
        if outcome is None:
            outcome = "ok" if rc == 0 else f"FAILED rc={rc}"
        return self.finish(j, t0, outcome, logp)

    def finish(self, j, t0, outcome, logp):
        t1 = now()
        r = {k: v for k, v in j.items() if not k.startswith("_")}
        r.update(started=t0.isoformat(timespec="seconds"), ended=t1.isoformat(timespec="seconds"),
                 ran_min=round((t1 - t0).total_seconds() / 60, 1), outcome=outcome, log=str(logp))
        write_json(self.jobsdir / f"{j['seq']}-{j['name']}.json", r)
        if outcome == "stopped":      # the owner needed the machine back; it runs again next time, in its old slot
            QUEUE.mkdir(parents=True, exist_ok=True)
            Path(j["_file"]).replace(QUEUE / Path(j["_file"]).name)
            r["outcome"] = outcome = "stopped (re-queued)"
            write_json(self.jobsdir / f"{j['seq']}-{j['name']}.json", r)
        Path(j["_file"]).unlink(missing_ok=True)
        STATE.unlink(missing_ok=True)
        self.results.append(r)
        self.log(f"{j['name']}: {outcome} after {fmt_min(r['ran_min'])} (est {fmt_min(j['est_min'])})")
        self.summary()
        return r

    def summary(self, final=False):
        rs = sorted(self.results, key=lambda r: r["started"])
        lines = [f"# Night run {self.rundir.name} — {platform.node()}", ""]
        state = "finished" if final else "IN PROGRESS"
        lines.append(f"Runner {state}. Started {self.started.strftime('%F %H:%M %Z')}, deadline "
                     f"{self.deadline.strftime('%H:%M')}" +
                     (f", ended {now().strftime('%H:%M')} ({fmt_min((now() - self.started).total_seconds() / 60)})."
                      if final else "."))
        ok = sum(1 for r in rs if r["outcome"] == "ok")
        lines += ["", f"{ok} of {len(rs)} job(s) ok.", "",
                  "| # | job | result | ran | est | queued at rev | log |", "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rs, 1):
            lines.append(f"| {i} | {r['name']} | {r['outcome']} | {fmt_min(r['ran_min'])} | {fmt_min(r['est_min'])} "
                         f"| {r.get('rev') or '?'}{' +dirty' if r.get('dirty') else ''} | `{r['log']}` |")
        still = [j for j in queued()]
        if final and still:
            lines += ["", "## Still queued (not started)", ""]
            for j in still:
                why = self.why_left.get(j["name"], "")
                lines.append(f"- {j['name']} (est {fmt_min(j['est_min'])}){': ' + why if why else ''}")
        bad = [r for r in rs if r["outcome"] != "ok"]
        if bad:
            lines += ["", "## Not ok — last 25 lines of each log", ""]
            for r in bad:
                try:
                    tail = Path(r["log"]).read_text(errors="replace").splitlines()[-25:]
                except Exception:
                    tail = ["(log unreadable)"]
                lines += [f"### {r['name']} — {r['outcome']}", "", "```", *tail, "```", ""]
        pick = [r for r in rs if r.get("by") or r.get("doc") or r.get("note")]
        if pick:
            lines += ["", "## Who picks these up", ""]
            for r in pick:
                bits = [x for x in (r.get("by"), r.get("doc"), r.get("note")) if x]
                lines.append(f"- {r['name']}: " + " · ".join(bits))
        (self.rundir / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    def run(self):
        PIDFILE.write_text(str(os.getpid()))
        signal.signal(signal.SIGTERM, self.on_term)
        signal.signal(signal.SIGINT, self.on_term)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        self.log(f"runner up, pid {os.getpid()}, deadline {self.deadline.strftime('%F %H:%M')}, queue {ROOT}")
        for f in sorted(RUNNING.glob("*.json")) if RUNNING.exists() else []:
            f.replace(QUEUE / f.name)
            self.log(f"re-queued {f.name}: it was mid-run when an earlier runner died")
        self.keep_awake()
        try:
            while not self.stop_now:
                if STOPFLAG.exists():
                    self.log("STOP flag found — not starting another job")
                    break
                jobs = queued()
                if not jobs:
                    self.log("queue empty")
                    break
                pick = None
                for j in jobs:
                    end = now() + dt.timedelta(minutes=j["est_min"])
                    if end <= self.deadline:
                        pick = j
                        break
                    self.why_left[j["name"]] = (f"would have ended ~{hm(end)}, after the {hm(self.deadline)} "
                                                f"deadline")
                if not pick:
                    self.log("nothing left fits before the deadline")
                    break
                self.settle()
                if self.stop_now or STOPFLAG.exists():
                    continue
                self.log(f"starting {pick['name']} (est {fmt_min(pick['est_min'])})")
                self.run_job(pick)
        finally:
            self.summary(final=True)
            STATE.unlink(missing_ok=True)
            STOPFLAG.unlink(missing_ok=True)
            if runner_pid() == os.getpid():
                PIDFILE.unlink(missing_ok=True)
            if self.awake and self.awake.poll() is None:
                self.awake.terminate()
            self.log(f"runner done — {self.rundir / 'SUMMARY.md'}")


def cmd_run(a):
    Runner(dt.datetime.fromisoformat(a.deadline), a.rundir).run()


# ---------------------------------------------------------------- status / stop / morning

def cmd_status(a):
    pid = runner_pid()
    st = read_json(STATE)
    if pid:
        print(f"runner going, pid {pid}")
    else:
        print("no runner going")
    if pid and st:
        started = dt.datetime.fromisoformat(st["started"])
        el = (now() - started).total_seconds() / 60
        print(f"  now running: {st['name']} for {fmt_min(el)} (est {fmt_min(st['est_min'])}, "
              f"timeout {fmt_min(st['timeout_min'])})")
        try:
            tail = Path(st["log"]).read_text(errors="replace").splitlines()[-3:]
            for line in tail:
                print(f"    | {line[:160]}")
        except Exception:
            pass
    if STOPFLAG.exists():
        print("  STOP requested — it stops after the current job")
    d = latest_run_dir()
    if d:
        rs = [read_json(f) for f in sorted((d / "jobs").glob("*.json"))] if (d / "jobs").exists() else []
        rs = [r for r in rs if r]
        if rs:
            print(f"  finished in {d.name}:")
            for r in sorted(rs, key=lambda r: r["started"]):
                print(f"    {r['name']:<34} {r['outcome']:<14} {fmt_min(r['ran_min'])}")
    jobs = queued()
    print(f"  still queued: {len(jobs)} job(s), {fmt_min(sum(j['est_min'] for j in jobs))}")


def cmd_stop(a):
    pid = runner_pid()
    if not pid:
        print("no runner going")
        return
    STOPFLAG.parent.mkdir(parents=True, exist_ok=True)
    STOPFLAG.write_text(now().isoformat(timespec="seconds") + "\n")
    if a.now:
        os.kill(pid, signal.SIGTERM)
        print(f"stopping now: the runner (pid {pid}) kills the current job and writes the summary")
    else:
        print("the runner stops after the current job (`night.py stop --now` to kill that too)")


def cmd_morning(a):
    d = latest_run_dir()
    if not d or not (d / "SUMMARY.md").exists():
        print("no night runs yet")
        return
    print((d / "SUMMARY.md").read_text(), end="")
    if runner_pid():
        print("\n(the runner is still going — `night.py status`)")


def main():
    p = argparse.ArgumentParser(prog="night.py", description="the overnight queue for long runs",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sp = p.add_subparsers(dest="sub", required=True)

    s = sp.add_parser("add", help="queue a job for tonight")
    s.add_argument("name")
    s.add_argument("--est", type=float, required=True, help="estimated runtime, minutes (required)")
    s.add_argument("--timeout", type=float, default=0, help="kill after this many minutes (default max(2x est, est+20))")
    s.add_argument("--priority", type=int, default=50, help="lower runs first (default 50)")
    s.add_argument("--by", help="who queued it and for what, e.g. 'vscode-claude, L1 arm64 fix'")
    s.add_argument("--doc", help="the task doc / record the result belongs in")
    s.add_argument("--note", help="anything the morning reader needs")
    s.add_argument("--cwd", help="run from here (default: the current directory)")
    s.add_argument("--force", action="store_true", help=f"allow --est over {MAX_EST_MIN} (owner's explicit ask only)")
    s.add_argument("--allow-duplicate", action="store_true",
                   help="queue a command that is already queued (a pre-registered repeat only)")
    s.set_defaults(fn=cmd_add)

    s = sp.add_parser("list", help="the queue, in run order")
    s.add_argument("--deadline", default=DEFAULT_DEADLINE)
    s.add_argument("--from", dest="frm", metavar="HH:MM", help="project from this start time (default: now)")
    s.set_defaults(fn=cmd_list)

    s = sp.add_parser("start", help="bedtime: detach and run the queue")
    s.add_argument("--deadline", default=DEFAULT_DEADLINE, help="don't START a job that would end after this")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_start)

    s = sp.add_parser("_run")
    s.add_argument("--deadline", required=True)
    s.add_argument("--rundir", required=True)
    s.set_defaults(fn=cmd_run)

    sp.add_parser("status", help="what is running, what finished").set_defaults(fn=cmd_status)

    s = sp.add_parser("stop", help="stop after the current job")
    s.add_argument("--now", action="store_true", help="kill the current job too")
    s.set_defaults(fn=cmd_stop)

    sp.add_parser("morning", help="print the latest SUMMARY.md").set_defaults(fn=cmd_morning)

    s = sp.add_parser("drop", help="remove a queued job")
    s.add_argument("name")
    s.set_defaults(fn=cmd_drop)

    argv, cmd = sys.argv[1:], []
    if argv[:1] == ["add"] and "--" in argv:     # everything after `--` is the job's command, verbatim
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    a = p.parse_args(argv)
    a.cmd = cmd
    ROOT.mkdir(parents=True, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
