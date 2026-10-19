#!/usr/bin/env python3
"""timing_lock.py — one timed run per box (TE9, docs/tasks/task-test-efficiency-2026-09.md).

WHY. Two timed runs on one box measure each other. "A process collision invalidated the first launch, and its numbers
are discarded" (a September record); Aug 27's contaminated sweep is why bench_peer.py's per-cell idle gate exists at all.
night.py serializes its own queue, but nothing stopped a daytime smoke, another session's harness, or a timed job in a
second checkout from overlapping it. The idle gate cannot: it sees load, not intent, and a run between two of its cells
is invisible to it until the damage is done.

WHAT. Every timed harness holds an exclusive fcntl.flock on ~/.goinfer-timing.lock for the length of its run. A second
one REFUSES by default, naming the holder (pid, command, start, label), or waits with --wait / the caller's wait_s. The
holder's children proceed: the holder exports GOINFER_TIMING_LOCK=<token>, and an acquire that finds its own token in
the lock file is a no-op. So a night.py job that runs a run-*.sh that runs bench_peer.py takes the lock once, not three
times. The kernel drops a flock when its holder exits, crashes or is killed, so a lock can never go stale.

It is cooperative: a `go test -bench` typed by hand does not take it. The harnesses and the run-*.sh wrapper do.

usage:
  timing_lock.py run [--wait SECONDS] [--label TEXT] -- CMD [ARGS...]   hold the lock while CMD runs (for run-*.sh)
  timing_lock.py status                                                who holds it, if anyone (exit 1 when held by
                                                                       a run other than this process's own holder)
  timing_lock.py check --tool NAME [--wait S] [--ignore-timing-lock]  for a day tool that loads the box (a build, a
                                                                       test run): exit 0 when it may go ahead, 75 when
                                                                       a timed run holds the lock (it names the holder)

library:
  from timing_lock import hold
  with hold("bench_peer.py out.json"):   # raises SystemExit naming the holder, if another run has it
      ...

GOINFER_TIMING_LOCK_FILE overrides the lock path (the tests use it); nothing else reads the environment.
"""
import argparse
import contextlib
import datetime as dt
import errno
import fcntl
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

TOKEN_ENV = "GOINFER_TIMING_LOCK"
POLL_S = 5


def lock_path():
    return Path(os.environ.get("GOINFER_TIMING_LOCK_FILE") or Path.home() / ".goinfer-timing.lock")


def read_holder(path=None):
    """The holder record in the lock file ({} when empty or unreadable). Advisory: the flock is the truth."""
    try:
        return json.loads((path or lock_path()).read_text() or "{}")
    except (OSError, ValueError):
        return {}


def describe(h):
    if not h:
        return "an unidentified process (the lock file carries no holder record)"
    started = h.get("start", "?")
    age = ""
    try:
        age = f", {int((time.time() - dt.datetime.fromisoformat(started).timestamp()) // 60)} min ago"
    except (TypeError, ValueError):
        pass
    label = f", label {h['label']!r}" if h.get("label") else ""
    return f"pid {h.get('pid', '?')} on {h.get('host', '?')}, `{h.get('cmd', '?')}`, started {started}{age}{label}"


def pid_alive(pid):
    """Whether a process with this pid exists on this host (a pid we may not signal still exists)."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def fd_holders(path=None):
    """(pid, cmdline) of every other process of ours that has the lock file open, from /proc/*/fd. This is who keeps a flock alive when the
    recorded holder is gone: a child that inherited the descriptor. Processes we cannot inspect are not listed. Read-only."""
    path = str(path or lock_path())
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if not any(_same_file(str(f), path) for f in (d / "fd").iterdir()):
                continue
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        out.append((int(d.name), cmd))
    return out


def _same_file(fdlink, path):
    try:
        return os.path.samefile(fdlink, path)
    except OSError:
        return False


def recorded_pid_state(h):
    """'alive', 'DEAD' or 'unknown' (another host, or no pid recorded) for the pid in a holder record."""
    pid = h.get("pid") if h else None
    if not isinstance(pid, int) or h.get("host") not in (None, platform.node()):
        return "unknown"
    return "alive" if pid_alive(pid) else "DEAD"


class Held(Exception):
    def __init__(self, holder, path):
        self.holder, self.path = holder, path
        super().__init__(f"another timed run holds the timing lock ({path}): {describe(holder)}")


class Lock:
    """An acquired (or inherited) timing lock. release() is idempotent."""

    def __init__(self, fd, path, token, inherited):
        self.fd, self.path, self.token, self.inherited = fd, path, token, inherited

    def release(self):
        if self.fd is None:
            return
        try:
            os.ftruncate(self.fd, 0)
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None
            if os.environ.get(TOKEN_ENV) == self.token:
                del os.environ[TOKEN_ENV]


def acquire(label="", wait_s=0.0, cmd=None):
    """Take the timing lock, or inherit it from a parent that holds it. Raises Held if another run holds it after
    waiting wait_s seconds (0 = refuse at once)."""
    path = lock_path()
    mine = os.environ.get(TOKEN_ENV)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError as e:
            if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                os.close(fd)
                raise
            holder = read_holder(path)
            if mine and holder.get("token") == mine:  # a parent of ours holds it: proceed under it
                os.close(fd)
                return Lock(None, path, mine, inherited=True)
            if time.monotonic() >= deadline:
                os.close(fd)
                raise Held(holder, path)
            time.sleep(min(POLL_S, max(0.1, deadline - time.monotonic())))
    token = f"{os.getpid()}-{time.time_ns()}"
    rec = {"pid": os.getpid(), "host": platform.node(), "label": label, "token": token,
           "cmd": cmd or shlex.join(sys.argv), "start": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    os.ftruncate(fd, 0)
    os.pwrite(fd, (json.dumps(rec) + "\n").encode(), 0)
    os.fsync(fd)
    os.environ[TOKEN_ENV] = token  # children inherit it
    return Lock(fd, path, token, inherited=False)


@contextlib.contextmanager
def hold(label="", wait_s=0.0):
    """acquire() as a context manager that exits the program, naming the holder, if the lock is taken."""
    try:
        lk = acquire(label=label, wait_s=wait_s)
    except Held as e:
        sys.exit(f"REFUSED: {e}. One timed run per box (TE9): a second run would measure the first. Wait for it "
                 f"to finish, or pass a wait if the harness has one.")
    try:
        yield lk
    finally:
        lk.release()


def cmd_run(a):
    if not a.cmd:
        sys.exit("timing_lock.py run: no command (usage: run [--wait S] [--label L] -- CMD ARGS...)")
    try:
        lk = acquire(label=a.label, wait_s=a.wait, cmd=shlex.join(a.cmd))
    except Held as e:
        print(f"REFUSED: {e}. One timed run per box (TE9).", file=sys.stderr)
        return 75  # EX_TEMPFAIL: try again later
    child = subprocess.Popen(a.cmd)  # inherits GOINFER_TIMING_LOCK through os.environ

    def forward(sig, _frame):
        with contextlib.suppress(ProcessLookupError):
            child.send_signal(sig)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, forward)
    try:
        return child.wait()
    finally:
        lk.release()


def held_by_other():
    """The holder record of a timed run that holds the lock and is not this process's own holder, else None. A run
    under the holder (its token in our environment) reads it as free, as acquire() does."""
    path = lock_path()
    if not path.exists():
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    except OSError:
        holder = read_holder(path)
        mine = os.environ.get(TOKEN_ENV)
        if mine and holder.get("token") == mine:
            return None
        return holder or {"label": "unidentified"}
    finally:
        os.close(fd)


def day_check(tool, wait_s=0.0, ignore=False):
    """Gate a day tool that loads the box (a build, a test run) on the lock: it must not run beside a timed run,
    which it would contaminate. Returns 0 to go ahead, 75 (EX_TEMPFAIL) to refuse; prints why either way."""
    deadline = time.monotonic() + max(0.0, wait_s)
    while (h := held_by_other()) is not None:
        if ignore:
            print(f"{tool}: WARNING: running beside a timed run, by --ignore-timing-lock: {describe(h)}. Its timings "
                  "from now until this finishes are contaminated; say so in its record.", file=sys.stderr)
            return 0
        if time.monotonic() >= deadline:
            print(f"{tool}: REFUSED: a timed run holds the timing lock and this would load the box under it: "
                  f"{describe(h)}. Wait for it (`--wait SECONDS`, or `python3 scripts/timing_lock.py status`); "
                  "`--ignore-timing-lock` overrides, for emergencies only.", file=sys.stderr)
            return 75
        time.sleep(min(POLL_S, max(0.1, deadline - time.monotonic())))
    return 0


def cmd_check(a):
    return day_check(a.tool, wait_s=a.wait, ignore=a.ignore_timing_lock)


def cmd_status(_a):
    path = lock_path()
    h = held_by_other()
    if h is not None:
        state = recorded_pid_state(h)
        print(f"HELD: {describe(h)} [recorded pid {state}]")
        if state == "DEAD":
            print("  the recorded holder is gone but the flock is still held: a process that inherited the lock descriptor is keeping it.")
        holders = fd_holders(path)
        if holders:
            print("  processes with the lock file open (the holder, its children, and runs waiting for it):")
            for pid, cmd in holders:
                print(f"    pid {pid}: {cmd[:200]}")
        elif state == "DEAD":
            print("  no process we can inspect has the file open (one of another user's, or a mapping we cannot read).")
        if state == "DEAD":
            print("  nothing is cleared or deleted here: end the process listed above (a person's call), or wait; the kernel drops the flock when it exits.")
        return 1
    mine = os.environ.get(TOKEN_ENV)
    holder = read_holder(path) if path.exists() else {}
    if mine and holder.get("token") == mine:
        print(f"free to this run (inherited from {describe(holder)})")
    elif holder and recorded_pid_state(holder) == "DEAD":
        print(f"free ({path}; stale record left by a dead holder, pid {holder.get('pid')}: {describe(holder)}; the flock is not held, "
              "so the next run overwrites it)")
    else:
        print(f"free ({path}{'' if path.exists() else ' does not exist'})")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run", help="hold the lock while CMD runs")
    r.add_argument("--wait", type=float, default=0.0, help="seconds to wait for a holder to finish (default: refuse)")
    r.add_argument("--label", default="", help="shown to a refused contender")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status", help="who holds the lock")
    c = sub.add_parser("check", help="exit 0 if a day tool may load the box now, 75 if a timed run holds the lock")
    c.add_argument("--tool", default="this tool", help="named in the refusal")
    c.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the holder to finish (default: refuse)")
    c.add_argument("--ignore-timing-lock", action="store_true", help="go ahead anyway, with a warning (emergencies)")
    a = p.parse_args(argv)
    if a.sub == "run":
        if a.cmd and a.cmd[0] == "--":
            a.cmd = a.cmd[1:]
        return cmd_run(a)
    if a.sub == "check":
        return cmd_check(a)
    return cmd_status(a)


if __name__ == "__main__":
    sys.exit(main())
