#!/usr/bin/env python3
"""KV5-7d's MC3 control (docs/tasks/task-kv-architecture-2026-10.md §6, KV5-7's registration, G2 (7d)): KV5-6's MC3 cell, unchanged, new serve against old.
4 clients each on its own resident KV slot (-kv-sessions 4), bench_w7_plain.py, the 1.5B, 6 turns of 128 greedy tokens, --fixed-nonce, a fresh
server per cell, cells old new new old old new; decode's bands on the median of the 3 new/old pairs of aggregate tok/s, and every turn's content_sha
equal. Battery no longer voids a cell for this reading (amended 2026-11-15, recorded in the registration before any cell ran): each cell records its
power source and battery percentage. A finished cell is skipped on a re-run.

  run_mc3_kv5_7.py OLD_SERVE NEW_SERVE SCRIPTS_DIR OUT_DIR"""
import os, subprocess, sys, time

OLD, NEW, SCRIPTS, OUT = sys.argv[1:5]
os.makedirs(f"{OUT}/serve-logs", exist_ok=True)


def say(s):
    print(time.strftime("%H:%M:%S ") + s, flush=True)


def power():
    return " | ".join(subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[:2])


say(f"old {OLD}, new {NEW}, -kv-sessions 4; {power()}")
for i, arm in enumerate(["old", "new", "new", "old", "old", "new"], 1):
    name, serve = f"mc3s4-{i}-{arm}", (OLD if arm == "old" else NEW)
    if os.path.exists(f"{OUT}/{name}.json"):
        say(f"== {name}: done before, skipped")
        continue
    say(f"== {name}; {power()}")
    with open(f"{OUT}/{name}.log", "w") as f:
        rc = subprocess.run(["python3", "-u", "bench_w7_plain.py", "--clients", "4", "--engines", "goinfer", "--backend", "metal", "--fixed-nonce",
                             "--serve-args", "-kv-sessions 4", "--server-log", f"{OUT}/serve-logs/{name}.log", f"{OUT}/{name}.json"],
                            cwd=SCRIPTS, env=dict(os.environ, GOINFER_SERVE_CPU=serve), stdout=f, stderr=subprocess.STDOUT).returncode
    say(f"== {name} rc={rc}; {power()}")
    if rc != 0:
        sys.exit(rc)
say("done")
