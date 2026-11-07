#!/usr/bin/env python3
"""KV5-6's G2 cells by day (docs/tasks/task-kv-architecture-2026-10.md, KV5's plan: G2 and G4 whole at KV5-6), new against old, on the
pinned binaries and harness under ~/goinfer-work/kv5-6: the 1.5B's decode at 128, 2,048, 3,900 and 8,000 and gemma3-1b's at 128, 2,048
and 3,900 (bench_peer.py, A B B A, one quad a cell, G4 as amended); MC3 at 4 clients on the 1.5B (bench_w7_plain.py, 6 turns of 128
greedy tokens, cells old new new old old new, every turn's content_sha compared); prefill TTFT on the 1.5B and the 7B at K = 512 and
3,900 (ttft_ab.py, three alternating pairs). The 7B's decode cells are the night job run-g2-kv5-6-7b.sh.
new: 0b2c9bd0 (KV5-6's product code with both review passes, main merged); old: f8632013 (main just before KV5-1's merge). The decode
steps (qwen15, deep15, gemma) ran on 2ffc982e, whose decode path the second review pass does not touch; MC3 and TTFT read 0b2c9bd0. Run it under the exclusive timing lock, on AC:
  python3 scripts/timing_lock.py run --wait 1800 --label kv5-6-g2-day -- python3 docs/measurements/kv5-metal-2026-11/run_g2_kv5_6_day.py
Each step that exists in the output directory is skipped, so a rerun resumes."""
import os, subprocess, sys, time

D = os.path.expanduser("~/goinfer-work/kv5-6")
NEW, OLD = f"{D}/serve-new", os.path.expanduser("~/goinfer-work/kv5-2-g2/serve-old")
SCRIPTS = f"{D}/new-0b2c9bd0/scripts"
TTFT = f"{D}/new-0b2c9bd0/docs/measurements/kv5-metal-2026-11/ttft_ab.py"
OUT = f"{D}/g2-day"


def say(s):
    print(time.strftime("%H:%M:%S ") + s, flush=True)


def run(name, cmd, env):
    out = f"{OUT}/{name}.json"
    if os.path.exists(out):
        say(f"== {name}: done before, skipped")
        return
    say(f"== {name}")
    with open(f"{OUT}/{name}.log", "w") as f:
        rc = subprocess.run(cmd + [out], cwd=SCRIPTS, env=dict(os.environ, **env), stdout=f, stderr=subprocess.STDOUT).returncode
    say(f"== {name} rc={rc}")
    if rc != 0:
        tail = open(f"{OUT}/{name}.log").read().strip().splitlines()[-1:]
        say(f"   {tail[0][:200] if tail else ''}")
        if os.path.exists(out):
            os.rename(out, out + ".failed")
        sys.exit(75 if any(l.startswith("REFUSED") for l in open(f"{OUT}/{name}.log")) else 1)


def main():
    for f in (NEW, OLD, f"{SCRIPTS}/bench_peer.py", f"{SCRIPTS}/bench_w7_plain.py", TTFT):
        if not os.path.exists(f):
            sys.exit(f"missing {f}")
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]
    if "AC Power" not in batt:
        sys.exit(f"not on AC power: {batt}")
    os.makedirs(f"{OUT}/serve-logs", exist_ok=True)
    say(f"new 0b2c9bd0, old f8632013; {batt}")
    peer = {"BENCH_DEPTH_BACKEND": "metal", "BENCH_ENGINES": "goinfer,goinfer_old", "BENCH_BACKENDS": "metal", "BENCH_REQUIRE_BACKEND": "1",
            "BENCH_SERVE_LOG_DIR": f"{OUT}/serve-logs", "BENCH_LOAD_TIMEOUT": "420", "GOINFER_SERVE_METAL": NEW, "GOINFER_SERVE_METAL_OLD": OLD,
            "BENCH_SWAPOUT_VOID_MB": "0", "BENCH_SWAP_VOID_STOP": "3", "BENCH_IDLE_WAIT": "180", "BENCH_ABBA": "1"}
    bp = ["python3", "-u", "bench_peer.py"]
    run("qwen15", bp, dict(peer, BENCH_MODELS="1.5B", BENCH_DEPTHS="2048,3900"))
    run("deep15", bp, dict(peer, BENCH_MODELS="1.5B", BENCH_PHASES="B", BENCH_DEPTHS="8000", BENCH_DEEP_CTX="8192"))
    run("gemma", bp, dict(peer, BENCH_MODELS="gemma3-1b", BENCH_DEPTHS="2048,3900"))
    for i, arm in enumerate(["old", "new", "new", "old", "old", "new"], 1):
        serve = OLD if arm == "old" else NEW
        run(f"mc3-{i}-{arm}", ["python3", "-u", "bench_w7_plain.py", "--clients", "4", "--engines", "goinfer", "--backend", "metal", "--fixed-nonce",
                              "--server-log", f"{OUT}/serve-logs/mc3-{i}-{arm}.log"], {"GOINFER_SERVE_CPU": serve})
    for mk in ("1.5B", "7B"):
        run(f"ttft-{mk}", ["python3", "-u", TTFT, OLD, NEW], {"TTFT_MODEL": mk, "TTFT_SCRIPTS": SCRIPTS})
    say("done")


if __name__ == "__main__":
    main()
