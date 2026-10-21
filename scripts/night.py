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
                                     (refuses a command already queued from the same directory, and a run plan that pins a
                                     serve build carrying the per-process swap guard yet sets swap_guard_mb without Francis's
                                     recorded approval: see check_plan)
  night.py list [--from HH:MM]       the queue in run order, with estimates and projected finish
  night.py start [--deadline HH:MM]  bedtime: detach and run the queue (default deadline 06:30)
  night.py status                    what is running now and what finished tonight
  night.py stop [--now]              finish the current job, then stop  (--now: kill it too)
  night.py morning                   print the latest night's SUMMARY.md
  night.py drop NAME                 remove a queued job
  night.py serve                     the always-on runner (B38), run by scripts/install-night-agent.sh's launchd agent or systemd
                                     user service: starts each job when its gates allow (below), holds no lock while idle
  night.py status [--json]           under serve the first line is "running <job> ...", "settling for <job> ..." (the lock held
                                     before the job starts) or "idle, N queued ..."; --json's phase is idle, settling or running,
                                     and a settling runner reads busy (idle false, running names the job)
  night.py pause | resume            serve: start no new job / start again (under serve `stop` is a pause; `stop --now` also
                                     kills the running job, which is re-queued)
  night.py update-runner             move the runner's pinned checkout (~/goinfer-runner) to origin/main; serve restarts on it
                                     when idle

serve starts a job when: its --not-before HH:MM has passed and its estimate ends by the deadline after it (a night-only job);
its retry time has passed; the box is not under heavy load (1-min load at most half the cores, free memory at least the job's
--fit-need-gb plus 2 GB); and the timing lock is free. It takes the lock, then waits SERVE_SETTLE_S with the box quiet, then
runs the before_queue step and the job under that lock. A job that exits 75 (its own check refused) is re-queued every
RETRY_BACKOFF_MIN until --retry-until, or RETRY_MAX tries, then recorded VOID-refused. It holds the machine awake while a job
runs and, on AC power, while a night-only job is due within AWAKE_AHEAD_H hours, and the day's SUMMARY.md says when it slept.
A night-only job whose night ends without it starting moves to the next night (its retry-until with it); SUMMARY.md lists it.

A job is a SELF-CONTAINED command, normally a docs/measurements/<campaign>/run-*.sh: pinned revs or
binaries, its own durable log/record paths, nothing interactive, no Claude session needed. It runs with
the environment of whoever ran `start`, in the directory it was queued from. Its stdout/stderr go to
<night dir>/runs/<date>/<NAME>.log in addition to whatever logs the job writes itself.

Everything lives under $NIGHT_DIR (default ~/goinfer-logs/night): queue/, runs/<date>/, runner.pid.
The queue is per machine — the Mac and nobara each have their own.

Before its first job the runner runs the box's `before_queue` command, if one is set in $NIGHT_HOOKS (default
~/.config/goinfer/night-hooks.toml, outside the repo), under the timing lock: for example
`before_queue = "/home/francis/tack-runs/serve/stop"`, which stops a server still running from the day. It must exit 0,
also when there is nothing to stop. If it fails, times out, or the file cannot be read, no job starts; SUMMARY.md says
which.

The same file sets the jobs' PATH once for the box: `path_prepend = ["/usr/local/go/bin"]` goes in front of the PATH every job
(and the before_queue command) runs with, and `require_on_path = ["go", "nvidia-smi"]` names tools that must be found on that
final PATH. `start` and `start --dry-run` check them; a missing tool refuses the start (no job starts, every job stays queued,
SUMMARY.md names what was missing), with no warning-only mode.
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
HOOKS = Path(os.environ.get("NIGHT_HOOKS", str(Path.home() / ".config" / "goinfer" / "night-hooks.toml"))).expanduser()
HOOK_TIMEOUT_S = int(os.environ.get("NIGHT_HOOK_TIMEOUT_S", "120"))
# The always-on runner (`serve`, docs/queue-engineering.md B38).
PAUSEFLAG = ROOT / "PAUSE"       # serve starts no new job while this exists (`pause` / `resume`)
SERVEFILE = ROOT / "serve.json"  # {"pid", "since"}: the runner going is the always-on one
RETRY_RC = 75                    # EX_TEMPFAIL: a job's own fit check refused; the runner re-queues it after a backoff
RETRY_BACKOFF_MIN = float(os.environ.get("NIGHT_RETRY_BACKOFF_MIN", "15"))
RETRY_MAX = int(os.environ.get("NIGHT_RETRY_MAX", "8"))  # attempts when a job sets no --retry-until
SERVE_POLL_S = float(os.environ.get("NIGHT_SERVE_POLL_S", "30"))
SERVE_SETTLE_S = float(os.environ.get("NIGHT_SERVE_SETTLE_S", "60"))  # the lock held, the box quiet, this long before a job starts
AWAKE_AHEAD_H = 12               # hold the machine awake when a night-only job is due within this many hours
HEADROOM_MB = 2048               # the 2026-10-19 rule: free memory under a job's fit need plus 2 GB is heavy load


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


# ---------------------------------------------------------------- the swap-guard plan check

# A serve build at or after dba04134 (PR #90) watches the server's own swap at GOINFER_SWAP_GUARD (512 MB by default) and system swap at a
# 4096 MB backstop. A run plan that also sets swap_guard_mb raises the own-swap trip to that value, and CLAUDE.md puts every guard threshold
# change behind Francis's word. A plan that pins such a build and sets the key is refused at `add` unless it records his approval:
# "swap_guard_approved_by": "Francis" and "swap_guard_approved_on": "YYYY-MM-DD". Plans that pin an older build are untouched: that build has the
# system-wide guard alone and needs the larger value to run.
GUARD_COMMIT = "dba04134"
GUARD_MARK = b"VmSwap of this process and its descendants"  # the per-process guard's banner text, present in a binary that carries it
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _carries_guard(plan, cwd):
    """True/False for whether the build the plan pins carries the per-process guard, None when that cannot be told (no binary on disk, no
    usable rev). The binary is read for the guard's banner text; without it, the pin's rev (`<serve>.rev`, else `commit`) is placed against
    GUARD_COMMIT in the repo's history."""
    serve = plan.get("serve")
    if serve and Path(serve).expanduser().is_file():
        with open(Path(serve).expanduser(), "rb") as f:
            tail = b""
            while chunk := f.read(8 << 20):
                if GUARD_MARK in tail + chunk:
                    return True
                tail = chunk[-len(GUARD_MARK):]
        return False
    rev = ""
    if serve and Path(str(serve) + ".rev").expanduser().is_file():
        rev = Path(str(serve) + ".rev").expanduser().read_text().strip()
    rev = rev or str(plan.get("commit") or "")
    if not re.fullmatch(r"[0-9a-f]{7,40}", rev):
        return None
    try:
        r = subprocess.run(["git", "-C", str(cwd), "merge-base", "--is-ancestor", GUARD_COMMIT, rev], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: True, 1: False}.get(r.returncode)


def check_plan(path, cwd):
    """The refusal message for a run plan that pins a build carrying the per-process swap guard and sets swap_guard_mb without recorded approval,
    else None (no key, an older pin, an unreadable pin, or an approved one)."""
    try:
        plan = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(plan, dict) or "swap_guard_mb" not in plan:
        return None
    if _carries_guard(plan, cwd) is not True:
        return None
    if plan.get("swap_guard_approved_by") == "Francis" and _DATE_RE.match(str(plan.get("swap_guard_approved_on", ""))):
        return None
    return (f"{path}: sets swap_guard_mb {plan['swap_guard_mb']} and pins a serve build that carries the per-process swap guard ({GUARD_COMMIT} or later), "
            f"where the default (512 MB on the server's own swap, 4096 MB system-wide as a backstop) applies. A change to a guard threshold needs Francis's "
            f"word (CLAUDE.md): drop the key, or record his approval in the plan as \"swap_guard_approved_by\": \"Francis\" and "
            f"\"swap_guard_approved_on\": \"YYYY-MM-DD\".")


def plan_files(cmd, cwd):
    """The run-plan JSON files a job's command names: a .json argument, or one a script argument names (`PLAN=$B/plan.json`, with the script's own
    simple VAR=value assignments and the environment expanded). A reference built any other way is not found."""
    out = []
    given = dict(t.split("=", 1) for t in cmd if re.match(r"^[A-Za-z_]\w*=", t))  # `env B=/dir bash run.sh`: the queued form
    for tok in cmd:
        p = Path(tok).expanduser()
        p = p if p.is_absolute() else Path(cwd) / p
        if not p.is_file():
            continue
        if p.suffix == ".json":
            out.append(p)
            continue
        try:
            text = p.read_text() if p.stat().st_size < (1 << 20) else ""
        except (OSError, UnicodeDecodeError):
            continue
        env = {**os.environ, **given}
        for m in re.finditer(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(\S+)", text, re.M):
            if m.group(1) not in given and not re.search(r"\$\{[^}]*[:?=+-]|\$\(", m.group(2)):  # a default or a command substitution is not a value
                env[m.group(1)] = _expand(m.group(2).strip("\"'"), env)
        for m in re.finditer(r"[^\s\"'()=;]+\.json", text):
            q = Path(_expand(m.group(0), env)).expanduser()
            if q.is_file() and q not in out:
                out.append(q)
    return out


def _expand(s, env):
    """$VAR and ${VAR} from env, a few rounds (a variable defined from another); an unknown name is left as written."""
    for _ in range(4):
        n = re.sub(r"\$(?:\{(\w+)\}|(\w+))", lambda m: env.get(m.group(1) or m.group(2), m.group(0)), s)
        if n == s:
            break
        s = n
    return s


def check_job_plans(cmd, cwd):
    """The refusal messages for every run plan the command names (see check_plan)."""
    return [m for m in (check_plan(p, cwd) for p in plan_files(cmd, cwd)) if m]


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
    if problems := check_job_plans(cmd, cwd):
        die("refusing to queue: " + " | ".join(problems))
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
    t = now()
    extra = {}
    if getattr(a, "not_before", None):
        extra["not_before"] = night_start(a.not_before, t).isoformat(timespec="seconds")
    if getattr(a, "retry_until", None):
        extra["retry_until"] = parse_deadline(a.retry_until, t).isoformat(timespec="seconds")
    if getattr(a, "fit_need_gb", None):
        extra["fit_need_gb"] = a.fit_need_gb
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
        **extra,
    }
    write_json(QUEUE / f"{seq}-{a.name}.json", job)
    if a.est > SPLIT_WARN_MIN:
        print(f"night: note — {a.est} min is over {SPLIT_WARN_MIN}; prefer jobs of 3 h or less so a failure "
              f"costs one job, not the night.", file=sys.stderr)
    jobs = queued()
    total = sum(j["est_min"] for j in jobs)
    print(f"queued {a.name!r} (est {fmt_min(a.est)}, timeout {fmt_min(job['timeout_min'])}, rev {job['rev'] or '?'}"
          f"{' +dirty' if job['dirty'] else ''}). Tonight's queue: {len(jobs)} job(s), {fmt_min(total)}.")
    if extra.get("not_before"):
        print(f"night-only: not before {extra['not_before']}")
    if extra.get("retry_until"):
        print(f"an exit {RETRY_RC} (its own check refused) re-queues it every {fmt_min(RETRY_BACKOFF_MIN)} until {extra['retry_until']}")


def hooks_config():
    """The box's hooks file as (dict, None); ({}, None) when there is none; ({}, why) when it cannot be read: a set but unreadable
    file must not be read as no settings."""
    if not HOOKS.exists():
        return {}, None
    try:
        import tomllib
        return tomllib.loads(HOOKS.read_text()), None
    except Exception as e:
        return {}, f"{HOOKS} is not readable TOML ({e})"


def before_queue_cmd():
    """The box's before_queue command from HOOKS: (command, None), (None, None) when none is set, or (None, why) when the
    file cannot be read: a set but unreadable hook must not be read as no hook."""
    cfg, err = hooks_config()
    if err:
        return None, err
    cmd = cfg.get("before_queue")
    if cmd is None:
        return None, None
    if not isinstance(cmd, str) or not shlex.split(cmd):
        return None, f"{HOOKS}: before_queue must be a non-empty command string"
    return cmd, None


def hook_list(key):
    """HOOKS' list of non-empty strings under key: (list, None), ([], None) when unset, or ([], why) when it is not such a list."""
    cfg, err = hooks_config()
    if err:
        return [], err
    v = cfg.get(key)
    if v is None:
        return [], None
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
        return [], f"{HOOKS}: {key} must be a list of non-empty strings"
    return v, None


def queue_path():
    """The PATH every job and the before_queue command runs with: HOOKS' path_prepend in front of the starter's PATH."""
    pre, _ = hook_list("path_prepend")
    return os.pathsep.join([*(str(Path(x).expanduser()) for x in pre), os.environ.get("PATH", "")])


def queue_env(**extra):
    """The environment a job (or the hook) runs with: the starter's, its PATH widened by path_prepend, plus extra."""
    return dict(os.environ, PATH=queue_path(), **extra)


def missing_required():
    """(tools HOOKS' require_on_path names that the final PATH does not find, None), or ([], why) when the settings cannot be read."""
    req, err = hook_list("require_on_path")
    if err:
        return [], err
    _, err = hook_list("path_prepend")
    if err:
        return [], err
    path = queue_path()
    return [t for t in req if not shutil.which(t, path=path)], None


def required_report():
    """One line per required tool for `start`/`--dry-run`: (lines, missing)."""
    req, _ = hook_list("require_on_path")
    miss, _ = missing_required()
    path = queue_path()
    return [f"  {t}: {shutil.which(t, path=path) or 'NOT FOUND'}" for t in req], miss


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

def write_refused_summary(start, jobs, miss):
    """SUMMARY.md for a start that was refused before any runner existed: what was missing, and that every job is still queued."""
    rundir = RUNS / start.strftime("%Y-%m-%d")
    rundir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Night run {rundir.name} — {platform.node()}", "",
             f"**Not started.** Required on PATH but not found: {', '.join(f'`{t}`' for t in miss)}.",
             f"No job was started. Fix `{HOOKS}` (`path_prepend`, `require_on_path`) or the PATH of the shell that starts the queue, then start again.", "",
             f"PATH searched: `{queue_path()}`", "", "## Still queued (not started)", ""]
    lines += [f"- {j['name']} (est {fmt_min(j['est_min'])})" for j in jobs]
    (rundir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


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
    hook, herr = before_queue_cmd()
    if herr:
        die(f"not started: {herr}")
    print(f"before the first job: {hook} (exit 0 or no job starts)" if hook else f"before the first job: nothing ({HOOKS} sets no before_queue)")
    pre, _ = hook_list("path_prepend")
    if pre:
        print(f"job PATH: {', '.join(pre)} in front of the starter's PATH")
    lines, miss = required_report()
    if lines:
        print("required on PATH:", *lines, sep="\n")
    _, rerr = missing_required()
    if rerr:
        die(f"not started: {rerr}")
    if miss:
        why = f"not started: required on PATH but not found: {', '.join(miss)} (searched {queue_path()}). Fix {HOOKS} (path_prepend / require_on_path) or the starting shell's PATH; every job stays queued."
        if not a.dry_run:
            write_refused_summary(start, jobs, miss)
        die(why)
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


# FAILURE_MARKERS are the lines that make a job FAILED even when it exits 0: a script that prints a failure and exits 0 anyway (a missing
# `exit 1` after a red step) otherwise reads ok in the morning summary. A line tagged "[expected]" is exempt: a plant's red run, which a
# job prints on purpose, carries that tag.
FAILURE_MARKERS = re.compile(r"^\s*--- FAIL|^FAIL\b|\bFAILED\b|\bVOID\b")


def failure_marker(logp, start):
    """The first line of the job's own output (logp from byte offset start) that carries a failure marker, or None."""
    try:
        with open(logp, "rb") as f:
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("=== END "):
            break
        if "[expected]" not in line and FAILURE_MARKERS.search(line):
            return line.strip()
    return None


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
        self.hook = None  # what before_queue did: {"cmd", "outcome", "ok", "secs", "tail"}
        self.retries = False  # serve: an exit of RETRY_RC re-queues the job (finish) instead of failing it
        self.slept = []  # serve: the sleeps it saw (note_sleep)

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

    def before_queue(self):
        """Runs the box's before_queue command (before_queue_cmd) once, before the first job, holding the timing lock so no
        timed run, or a server start that checks the lock, lands in between. Records what happened in self.hook and returns
        whether the queue may start."""
        cmd, err = before_queue_cmd()
        t0 = time.time()
        miss, rerr = missing_required()
        if err:
            self.hook = {"cmd": None, "ok": False, "outcome": f"FAILED: {err}", "secs": 0, "tail": []}
        elif rerr or miss:
            self.hook = {"cmd": None, "ok": False, "secs": 0, "tail": [],
                         "outcome": "FAILED: " + (rerr or f"required on PATH but not found: {', '.join(miss)}")}
        elif cmd is None:
            self.hook = {"cmd": None, "ok": True, "outcome": f"nothing to run ({HOOKS} sets no before_queue)", "secs": 0, "tail": []}
        else:
            try:
                lock = timing_lock.acquire(label="night.py before_queue", wait_s=LOCK_WAIT_MIN * 60, cmd=cmd)
            except timing_lock.Held as e:
                self.hook = {"cmd": cmd, "ok": False, "outcome": f"FAILED: timing lock held ({e})", "secs": 0, "tail": []}
            else:
                try:
                    r = subprocess.run(shlex.split(cmd), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                       timeout=HOOK_TIMEOUT_S, env=queue_env())
                    ok, outcome, out = r.returncode == 0, f"exit {r.returncode}", r.stdout + r.stderr
                except subprocess.TimeoutExpired as e:
                    ok, outcome = False, f"timed out after {HOOK_TIMEOUT_S} s"
                    out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                except OSError as e:
                    ok, outcome, out = False, f"could not run ({e})", ""
                finally:
                    lock.release()
                self.hook = {"cmd": cmd, "ok": ok, "outcome": outcome if ok else "FAILED: " + outcome,
                             "secs": round(time.time() - t0, 1), "tail": out.strip().splitlines()[-10:]}
        h = self.hook
        self.log(f"before the queue: {h['cmd'] or '-'}: {h['outcome']}" + (f" in {h['secs']} s" if h["cmd"] else ""))
        for line in h["tail"]:
            self.log(f"  | {line}")
        if not h["ok"]:
            for j in queued():
                self.why_left[j["name"]] = "not started: the before_queue step failed (the hook, or a required tool missing from PATH)"
        self.summary()
        return h["ok"]

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

    def claim(self, j):
        """Moves a picked job's file from queue/ to running/ and returns the job as that file holds it, or None if the file is
        no longer in queue/. The pick was read before settle(), so a drop or a re-add (a new file) in the meantime can take
        the file away; the move is the claim, a vanished file is skipped, and the loop reads the queue again, where a
        re-added job is picked from its new entry. The log says which happened."""
        src = Path(j["_file"])
        RUNNING.mkdir(parents=True, exist_ok=True)
        moved = RUNNING / src.name
        try:
            src.replace(moved)
        except FileNotFoundError:
            how = ("re-added, so it runs from its new entry" if any(q["name"] == j["name"] for q in queued())
                   else "dropped, so it does not run")
            self.log(f"{j['name']}: skipped, its queue file {src.name} left the queue before it started ({how})")
            return None
        c = read_json(moved)
        if not c:
            moved.replace(src)  # an unreadable file is not a job: queued() passes it over
            self.log(f"{j['name']}: skipped, its queue file {src.name} became unreadable before it started")
            return None
        c["_file"] = str(moved)
        return c

    def run_job(self, j, lock=None):
        """Runs one queued job. lock is a timing lock the caller already holds (serve takes it before settling); None: the job takes
        it here, waiting up to LOCK_WAIT_MIN."""
        j = self.claim(j)
        if j is None:
            return None
        name = j["name"]
        logp = self.rundir / f"{name}.log"
        cwd = j["cwd"]
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
            j["_out_from"] = f.tell()  # where the job's own output starts: the header above (its note, its cmd) is not scanned
            # TE9: the job holds the timing lock, and its children (a run-*.sh, bench_peer.py) inherit it through the
            # environment. A timed run still going from the day, or from another checkout, is waited for up to
            # LOCK_WAIT_MIN; then this job is skipped, naming the holder, rather than measured on top of it.
            own = lock is None
            if own:
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
                if own:
                    lock.release()

    def _run_locked(self, j, name, cwd, t0, logp, f):
        """The job itself: run_job calls this with the log open and the timing lock held."""
        cmd = j["cmd"] if len(j["cmd"]) > 1 else ["/bin/bash", "-c", j["cmd"][0]]
        env = queue_env(NIGHT_JOB=name, NIGHT_LOG=str(logp), NIGHT_RUNDIR=str(self.rundir))
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
            outcome = "ok" if rc == 0 else ("retry" if rc == RETRY_RC and self.retries else f"FAILED rc={rc}")
        if outcome == "ok":
            bad = failure_marker(logp, j.get("_out_from", 0))
            if bad:
                outcome = f"FAILED rc=0, log says: {bad[:120]}"
        return self.finish(j, t0, outcome, logp)

    def finish(self, j, t0, outcome, logp):
        t1 = now()
        r = {k: v for k, v in j.items() if not k.startswith("_")}
        r.update(started=t0.isoformat(timespec="seconds"), ended=t1.isoformat(timespec="seconds"),
                 ran_min=round((t1 - t0).total_seconds() / 60, 1), outcome=outcome, log=str(logp))
        write_json(self.jobsdir / f"{j['seq']}-{j['name']}.json", r)
        if outcome == "retry":
            # The job's own check refused (exit RETRY_RC): back to the queue after a backoff, until its --retry-until (or RETRY_MAX tries).
            attempts = j.get("attempts", 0) + 1
            nxt = t1 + dt.timedelta(minutes=RETRY_BACKOFF_MIN)
            until = dt.datetime.fromisoformat(j["retry_until"]) if j.get("retry_until") else None
            if (until and nxt > until) or (not until and attempts >= RETRY_MAX):
                r["outcome"] = outcome = (f"VOID-refused: its check refused {attempts} time(s), "
                                          + (f"retry-until {hm(until)}" if until else f"{RETRY_MAX} tries"))
            else:
                q = {k: v for k, v in j.items() if not k.startswith("_")}
                q.update(attempts=attempts, next_try=nxt.isoformat(timespec="seconds"))
                QUEUE.mkdir(parents=True, exist_ok=True)
                write_json(QUEUE / Path(j["_file"]).name, q)
                r["outcome"] = outcome = f"retry {attempts} (its check refused; next at {hm(nxt)})"
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
        if self.hook:
            h = self.hook
            lines += ["", f"Before the queue: `{h['cmd']}` {h['outcome']}" + (f" in {h['secs']} s." if h["cmd"] else ".") if h["cmd"]
                      else f"Before the queue: {h['outcome']}."]
            if not h["ok"]:
                lines.append("**No job was started.** Fix the command (or the hooks file) and start again.")
            if h["tail"]:
                lines += ["", "```", *h["tail"], "```"]
        missed = []
        try:
            missed = [json.loads(x) for x in (self.rundir / "missed.jsonl").read_text().splitlines() if x.strip()]
        except (OSError, ValueError):
            pass
        if missed:
            lines += ["", "## Missed its night (moved to the next)", ""]
            lines += [f"- {m['name']}: the night {m['night']} ended without a start; last wait: {m['last_wait']}. Now not before "
                      f"{m['moved_to']}." for m in missed]
        if self.slept:
            lines += ["", "**The machine slept:** " + "; ".join(self.slept) + ". A job running across a sleep was timed over it."]
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
                    wait = max((dt.datetime.fromisoformat(j[k]) for k in ("not_before", "next_try") if j.get(k)), default=None)
                    if wait and wait > now():
                        self.why_left[j["name"]] = f"not before {hm(wait)}"
                        continue
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
                if self.hook is None and not self.before_queue():
                    break
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


# ---------------------------------------------------------------- the always-on runner (serve, B38)

def fcntl_flock(fd):
    """An exclusive flock on fd, for a test standing in another holder of the timing lock."""
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def free_mem_mb():
    """This box's available memory in MiB, the number goinfer's fit guard reads (scripts/hostmem.py, cmd/hostmem). None when
    unreadable."""
    import hostmem
    return hostmem.available_mb()


def night_start(hhmm, added, deadline=DEFAULT_DEADLINE):
    """When a job with --not-before hhmm, added at `added`, may first start: hhmm on the night that added falls in, or the next one. A
    night runs from hhmm to the deadline the next morning; a job added inside one may start at once."""
    h, m = (int(x) for x in hhmm.split(":"))
    dh, dm = (int(x) for x in deadline.split(":"))
    for back in (1, 0):
        start = (added - dt.timedelta(days=back)).replace(hour=h, minute=m, second=0, microsecond=0)
        end = start.replace(hour=dh, minute=dm)
        if end <= start:
            end += dt.timedelta(days=1)
        if start <= added < end:
            return start
    start = added.replace(hour=h, minute=m, second=0, microsecond=0)
    return start if start > added else start + dt.timedelta(days=1)


def night_end(start, deadline=DEFAULT_DEADLINE):
    """The deadline that closes the night opening at start."""
    dh, dm = (int(x) for x in deadline.split(":"))
    end = start.replace(hour=dh, minute=dm, second=0, microsecond=0)
    return end if end > start else end + dt.timedelta(days=1)


class Server(Runner):
    """The always-on runner: a loop over the queue that starts each job when its gates allow (eligible) and the timing lock is free. It
    takes the lock before settling, so no other timed run starts between a job's settle and its start, and holds nothing while idle, so
    the day tools run between jobs. A job's own refusal (exit RETRY_RC) re-queues it after a backoff. One SUMMARY.md per day."""

    def __init__(self):
        t = self.clock()
        super().__init__(night_end(t), RUNS / t.strftime("%Y-%m-%d"))
        self.retries = True
        self.awake_proc = None
        self.exit_when_idle = False
        self.hook_retry_at = None  # a failed before_queue is tried again from here, not every poll
        self.last, self.waits, self.since = None, [], None  # what the last pass did and why each job waits, for status
        self.phase, self.job = "idle", None  # "settling" (the lock held, before a job) or "running", and that job
        self.last_why = {}  # each queued job's last reason to wait, for a missed night's record
        self.drift0 = None  # wall clock minus monotonic at the last pass: it grows by however long the machine slept

    # The box: methods so a test can stand in a clock, a load and a memory reading.
    def clock(self):
        return now()

    def load1(self):
        return os.getloadavg()[0]

    def ncores(self):
        return os.cpu_count() or 1

    def free_mb(self):
        return free_mem_mb()

    def heavy(self, j):
        """Why the box is under heavy load for job j now (the 2026-10-19 rule), or None."""
        la, n = self.load1(), self.ncores()
        if la > n / 2:
            return f"load {la:.2f} above half of {n} cores"
        need = float(j.get("fit_need_gb") or 0) * 1024
        free = self.free_mb()
        if need and free is not None and free < need + HEADROOM_MB:
            return f"free memory {free} MB under the job's fit need {need:.0f} MB + {HEADROOM_MB} MB"
        return None

    def eligible(self, j, t):
        """Why j may not start at t, or None."""
        if j.get("not_before") and dt.datetime.fromisoformat(j["not_before"]) > t:
            return f"night-only: not before {hm(dt.datetime.fromisoformat(j['not_before']))}"
        if j.get("next_try") and dt.datetime.fromisoformat(j["next_try"]) > t:
            return f"retry at {hm(dt.datetime.fromisoformat(j['next_try']))}"
        if j.get("not_before"):
            end = night_end(dt.datetime.fromisoformat(j["not_before"]))
            if t + dt.timedelta(minutes=j["est_min"]) > end:
                return f"would end after the {hm(end)} deadline"
        return self.heavy(j)

    def roll_missed(self, j, t):
        """A night-only job whose night ended without it starting moves to the next night: its not-before, and its retry-until by the
        same shift, with its retry count cleared. Recorded in the day's missed.jsonl, which SUMMARY.md lists. Returns whether it moved."""
        nb = dt.datetime.fromisoformat(j["not_before"])
        end = night_end(nb)
        if t < end:
            return False
        new = night_start(nb.strftime("%H:%M"), t)
        shift = new - nb
        q = {k: v for k, v in j.items() if not k.startswith("_")}
        q["not_before"] = new.isoformat(timespec="seconds")
        if q.get("retry_until"):
            q["retry_until"] = (dt.datetime.fromisoformat(q["retry_until"]) + shift).isoformat(timespec="seconds")
        q.pop("next_try", None)
        q["attempts"] = 0
        q["missed_nights"] = j.get("missed_nights", 0) + 1
        write_json(Path(j["_file"]), q)
        why = self.last_why.get(j["name"], "no pass saw it")
        rec = {"name": j["name"], "night": f"{nb.isoformat(timespec='minutes')} to {end.isoformat(timespec='minutes')}",
               "last_wait": why, "moved_to": q["not_before"]}
        with open(self.rundir / "missed.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.log(f"{j['name']}: its night ({hm(nb)}-{hm(end)}) ended without a start (last wait: {why}); moved to {q['not_before']}")
        self.summary()
        return True

    def settle_locked(self, j):
        """Waits, holding the lock, for the box to stay quiet for SERVE_SETTLE_S. Returns why it did not, or None."""
        waited = 0.0
        while waited < SERVE_SETTLE_S:
            if self.stop_now or PAUSEFLAG.exists():
                return "paused"
            if why := self.heavy(j):
                return why
            step = min(5.0, SERVE_SETTLE_S - waited)
            self.sleep(step)
            waited += step
        return None

    def sleep(self, s):
        time.sleep(s)

    def drift(self):
        """Wall clock minus the monotonic clock, which stops while the machine sleeps on darwin and Linux alike."""
        return time.time() - time.monotonic()

    def note_sleep(self, t):
        """Records a sleep since the last pass (one over a minute) in the day's summary: a job that ran across it was timed over it."""
        d = self.drift()
        if self.drift0 is not None and d - self.drift0 > 60:
            msg = f"the machine slept about {fmt_min((d - self.drift0) / 60)}, waking before {hm(t)}"
            self.slept.append(msg)
            self.log(msg)
            self.summary()
        self.drift0 = d

    def day(self, t):
        """Moves the runner's record to t's day: a new rundir and SUMMARY.md each day."""
        d = RUNS / t.strftime("%Y-%m-%d")
        if d != self.rundir:
            self.rundir, self.jobsdir = d, d / "jobs"
            self.jobsdir.mkdir(parents=True, exist_ok=True)
            self.results = [r for r in (read_json(f) for f in sorted(self.jobsdir.glob("*.json"))) if r]
            self.started, self.deadline, self.slept = t, night_end(t), []

    def on_ac(self):
        if sys.platform != "darwin":
            return True
        try:
            return "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return True

    def awake_wanted(self, t):
        """Whether to hold the machine awake while idle: when a night-only job is due within AWAKE_AHEAD_H hours and the box is on AC
        power (a Mac on battery that never sleeps drains). A running job always holds it (tick)."""
        if not self.on_ac():
            return False
        for j in queued():
            nb = j.get("not_before")
            if nb and dt.datetime.fromisoformat(nb) <= t + dt.timedelta(hours=AWAKE_AHEAD_H):
                return True
        return False

    def hold_awake(self, on):
        if on and (self.awake_proc is None or self.awake_proc.poll() is not None):
            me = str(os.getpid())
            if sys.platform == "darwin" and shutil.which("caffeinate"):
                cmd = ["caffeinate", "-i", "-s", "-w", me]
            elif shutil.which("systemd-inhibit"):
                cmd = ["systemd-inhibit", "--what=sleep:idle", "--who=goinfer-night", "--why=a queued night job",
                       "--mode=block", "/bin/sh", "-c", f"while kill -0 {me} 2>/dev/null; do sleep 30; done"]
            else:
                return
            try:
                self.awake_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.log(f"holding the machine awake: {cmd[0]}")
            except OSError as e:
                self.log(f"WARNING: could not start {cmd[0]}: {e}")
        elif not on and self.awake_proc is not None and self.awake_proc.poll() is None:
            self.awake_proc.terminate()
            self.awake_proc = None
            self.log("no night job due: the machine may sleep")

    def tick(self):
        """One pass: starts at most one job. Returns what happened: "paused", "idle", "lock held", "settle: <why>" or "ran"."""
        t = self.clock()
        self.day(t)
        self.note_sleep(t)
        if PAUSEFLAG.exists():
            return "paused"
        self.hold_awake(self.awake_wanted(t))
        pick, waits = None, []
        for j in queued():
            if j.get("not_before") and self.roll_missed(j, t):
                continue
            why = self.eligible(j, t)
            self.last_why[j["name"]] = why
            if why is None:
                pick = j
                break
            waits.append(f"{j['name']}: {why}")
        self.waits = waits
        if pick is None:
            return "idle"
        if self.hook_retry_at and t < self.hook_retry_at:
            return f"before_queue failed; tried again at {hm(self.hook_retry_at)}"
        try:
            lock = timing_lock.acquire(label=f"night.py serve: settling for {pick['name']}", wait_s=0, cmd="night.py serve")
        except timing_lock.Held:
            return "lock held"
        try:
            self.last = f"settling for {pick['name']}"
            self.phase, self.job = "settling", pick["name"]
            self.publish()
            if why := self.settle_locked(pick):
                return f"settle: {why}"
            # The box's before_queue step (a day server's stop) runs before every job: a server started between jobs holds the
            # memory a job's fit check counts. It runs under the lock taken above.
            if not self.before_queue():
                self.hook_retry_at = t + dt.timedelta(minutes=RETRY_BACKOFF_MIN)
                return "before_queue failed"
            self.hook_retry_at = None
            self.hold_awake(True)
            self.log(f"starting {pick['name']} (est {fmt_min(pick['est_min'])})")
            self.phase = "running"
            self.publish()
            self.run_job(pick, lock=lock)
            return "ran"
        finally:
            self.phase, self.job = "idle", None
            lock.release()

    def publish(self):
        """SERVEFILE: who the runner is and what its last pass did, for status."""
        if self.since is None:
            return  # tick() outside serve(): a test
        write_json(SERVEFILE, {"pid": os.getpid(), "since": self.since, "last": self.last, "waits": self.waits,
                               "phase": self.phase, "job": self.job, "at": self.clock().isoformat(timespec="seconds")})

    def serve(self):
        PIDFILE.write_text(str(os.getpid()))
        self.since, self.waits = self.clock().isoformat(timespec="seconds"), []
        self.publish()
        signal.signal(signal.SIGTERM, self.on_term)
        signal.signal(signal.SIGINT, self.on_term)
        signal.signal(signal.SIGHUP, lambda *_: setattr(self, "exit_when_idle", True))
        self.log(f"serve up, pid {os.getpid()}, queue {ROOT}")
        for f in sorted(RUNNING.glob("*.json")) if RUNNING.exists() else []:
            f.replace(QUEUE / f.name)
            self.log(f"re-queued {f.name}: it was mid-run when an earlier runner died")
        last = None
        try:
            while not self.stop_now and not self.exit_when_idle:
                what = self.tick()
                if what != last and what != "ran":
                    self.log(f"serve: {what}" + (f" ({'; '.join(self.waits)})" if what == "idle" and self.waits else ""))
                last = self.last = what
                self.publish()
                if what != "ran":
                    self.sleep(SERVE_POLL_S)
        finally:
            self.hold_awake(False)
            STATE.unlink(missing_ok=True)
            if runner_pid() == os.getpid():
                PIDFILE.unlink(missing_ok=True)
                SERVEFILE.unlink(missing_ok=True)
            self.log("serve down")


def serve_pid():
    """The always-on runner's pid when it is the runner going, else None."""
    pid, sv = runner_pid(), read_json(SERVEFILE)
    return pid if pid and sv and sv.get("pid") == pid else None


def cmd_serve(a):
    if runner_pid():
        die(f"a runner is already going (pid {runner_pid()})")
    Server().serve()


def status_json():
    """The runner as data, for a launcher that must wait for it: {"runner": "serve"|"night"|None, "running": <job>|None, "phase":
    "idle"|"settling"|"running"|None, ...}. A settling runner holds the timing lock for its job: running names it and idle is false."""
    pid, st = runner_pid(), read_json(STATE)
    kind = ("serve" if serve_pid() else "night") if pid else None
    jobs = queued()
    sv = read_json(SERVEFILE) if kind == "serve" else None
    running_job = (st or {}).get("name") if pid else None
    phase = "running" if running_job else "idle" if pid else None
    if not running_job and sv and sv.get("phase") == "settling":
        running_job, phase = sv.get("job"), "settling"  # the lock is held for this job: busy, though it has not started
    return {"runner": kind, "pid": pid, "paused": PAUSEFLAG.exists(), "running": running_job, "idle": running_job is None,
            "phase": phase, "last": (sv or {}).get("last"), "waits": (sv or {}).get("waits") or [],
            "queued": [j["name"] for j in jobs], "lock_free": timing_lock.held_by_other() is None}


def cmd_pause(a):
    PAUSEFLAG.parent.mkdir(parents=True, exist_ok=True)
    PAUSEFLAG.write_text(now().isoformat(timespec="seconds") + "\n")
    print("paused: no new job starts (a running one finishes; `night.py resume` to go on)")


def cmd_resume(a):
    PAUSEFLAG.unlink(missing_ok=True)
    print("resumed")


RUNNER_DIR = Path(os.environ.get("NIGHT_RUNNER_DIR", str(Path.home() / "goinfer-runner"))).expanduser()


def cmd_update_runner(a):
    """Moves the pinned runner worktree (RUNNER_DIR, the checkout the service runs night.py from) to origin/main and tells serve to
    restart on it once idle. Nobody edits RUNNER_DIR: a job runs from the directory it was queued from, not from here."""
    repo = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "-C", str(repo), "fetch", "-q", "origin", "main"]).returncode:
        die("git fetch failed")
    if not RUNNER_DIR.exists():
        if subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(RUNNER_DIR), "origin/main"]).returncode:
            die(f"could not add the runner worktree at {RUNNER_DIR}")
    else:
        if git(RUNNER_DIR, "status", "--porcelain", "--untracked-files=no"):
            die(f"{RUNNER_DIR} has local edits; it is the pinned runner checkout and nobody edits it. Look, then `git -C {RUNNER_DIR} checkout .`")
        if subprocess.run(["git", "-C", str(RUNNER_DIR), "checkout", "-q", "--detach", "origin/main"]).returncode:
            die("checkout failed")
    print(f"{RUNNER_DIR} at {git(RUNNER_DIR, 'rev-parse', '--short', 'HEAD')}")
    pid = serve_pid()
    if pid:
        os.kill(pid, signal.SIGHUP)
        print(f"serve (pid {pid}) exits when idle and its service manager restarts it on this code")


# ---------------------------------------------------------------- status / stop / morning

def cmd_status(a):
    if a.json:
        print(json.dumps(status_json()))
        return
    pid = runner_pid()
    st = read_json(STATE)
    sv = read_json(SERVEFILE) if serve_pid() else None
    if sv:
        # The first line starts with the state, "running <job>" or "idle"/"paused": a launcher reads a box as busy from that prefix.
        who = f"(serve, pid {pid}, up since {sv.get('since')})"
        if st:
            print(f"running {st['name']} {who}")
        elif sv.get("phase") == "settling":
            print(f"settling for {sv.get('job')}, the timing lock held {who}")
        else:
            print(f"{'paused' if PAUSEFLAG.exists() else 'idle'}, {len(queued())} queued {who}: last pass {sv.get('last') or 'none yet'}")
        if not st:
            for w in sv.get("waits") or []:
                print(f"  waits: {w}")
    elif pid:
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
    if PAUSEFLAG.exists():
        print("  PAUSED — no new job starts (`night.py resume`)")
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
    if serve_pid():
        # The always-on runner is restarted by its service manager, so a stop is a pause: no new job until `resume`.
        cmd_pause(a)
        if a.now:
            os.kill(pid, signal.SIGTERM)
            print(f"and the current job is killed and re-queued (serve, pid {pid}, restarts paused)")
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
    s.add_argument("--not-before", metavar="HH:MM", help="night-only: start no earlier than this, and end by the deadline after it")
    s.add_argument("--retry-until", metavar="HH:MM", help=f"an exit {RETRY_RC} (the job's own check refused) re-queues it until then")
    s.add_argument("--fit-need-gb", type=float, help="memory the job needs; serve starts it only with this plus 2 GB free")
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

    s = sp.add_parser("status", help="what is running, what finished")
    s.add_argument("--json", action="store_true", help="the runner as one JSON object (runner, running, idle, queued, ...)")
    s.set_defaults(fn=cmd_status)

    sp.add_parser("serve", help="the always-on runner (run by the launchd agent / systemd user service)").set_defaults(fn=cmd_serve)
    sp.add_parser("pause", help="serve: start no new job").set_defaults(fn=cmd_pause)
    sp.add_parser("resume", help="serve: start jobs again").set_defaults(fn=cmd_resume)
    sp.add_parser("update-runner", help="move the pinned runner checkout to origin/main; serve restarts on it when idle").set_defaults(fn=cmd_update_runner)

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
