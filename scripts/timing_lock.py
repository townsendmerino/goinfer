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
  timing_lock.py status                                                who holds it, if anyone

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


def cmd_status(_a):
    path = lock_path()
    if not path.exists():
        print(f"free ({path} does not exist)")
        return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        print(f"free ({path})")
        return 0
    except OSError:
        print(f"HELD: {describe(read_holder(path))}")
        return 1
    finally:
        os.close(fd)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="sub", required=True)
    r = sub.add_parser("run", help="hold the lock while CMD runs")
    r.add_argument("--wait", type=float, default=0.0, help="seconds to wait for a holder to finish (default: refuse)")
    r.add_argument("--label", default="", help="shown to a refused contender")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status", help="who holds the lock")
    a = p.parse_args(argv)
    if a.sub == "run":
        if a.cmd and a.cmd[0] == "--":
            a.cmd = a.cmd[1:]
        return cmd_run(a)
    return cmd_status(a)


if __name__ == "__main__":
    sys.exit(main())
