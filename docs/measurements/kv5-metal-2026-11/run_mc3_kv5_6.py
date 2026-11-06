#!/usr/bin/env python3
"""KV5-6's G2 MC3 cell as KV5's plan registers it, 4 clients each on its own resident KV slot: serve's default is now 2 slots (Metal keeps
2, E-P09), and at the default 4 clients thrash 2 slots and re-prefill nearly every turn, which is not the batched decode the cell reads.
So every cell passes -kv-sessions 4. bench_w7_plain.py, the 1.5B, 6 turns of 128 greedy tokens, --fixed-nonce, a fresh server per cell,
cells old new new old old new; decode's bands on the median of the 3 new/old pairs of aggregate tok/s, and every turn's content_sha equal.
new: serve-new at 0b2c9bd0; old: f8632013. Run under the exclusive timing lock, on AC:
  python3 scripts/timing_lock.py run --wait 1800 --label kv5-6-mc3s4 -- python3 docs/measurements/kv5-metal-2026-11/run_mc3_kv5_6.py"""
import os, subprocess, sys, time

D = os.path.expanduser("~/goinfer-work/kv5-6")
NEW, OLD = f"{D}/serve-new", os.path.expanduser("~/goinfer-work/kv5-2-g2/serve-old")
SCRIPTS = f"{D}/new-0b2c9bd0/scripts"
OUT = f"{D}/g2-day"


def say(s):
    print(time.strftime("%H:%M:%S ") + s, flush=True)


batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]
if "AC Power" not in batt:
    sys.exit(f"not on AC power: {batt}")
say(f"new 0b2c9bd0, old f8632013, -kv-sessions 4; {batt}")
for i, arm in enumerate(["old", "new", "new", "old", "old", "new"], 1):
    name, serve = f"mc3s4-{i}-{arm}", (OLD if arm == "old" else NEW)
    if os.path.exists(f"{OUT}/{name}.json"):
        say(f"== {name}: done before, skipped")
        continue
    say(f"== {name}")
    with open(f"{OUT}/{name}.log", "w") as f:
        rc = subprocess.run(["python3", "-u", "bench_w7_plain.py", "--clients", "4", "--engines", "goinfer", "--backend", "metal", "--fixed-nonce",
                             "--serve-args", "-kv-sessions 4", "--server-log", f"{OUT}/serve-logs/{name}.log", f"{OUT}/{name}.json"],
                            cwd=SCRIPTS, env=dict(os.environ, GOINFER_SERVE_CPU=serve), stdout=f, stderr=subprocess.STDOUT).returncode
    say(f"== {name} rc={rc}")
    if rc != 0:
        sys.exit(rc)
say("done")
