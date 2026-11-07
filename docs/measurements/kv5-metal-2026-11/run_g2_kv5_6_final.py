#!/usr/bin/env python3
"""KV5-6's timed cells on the final build, by day, each step its own hold of the exclusive timing lock (so another session can take a
turn between them), on AC. new: serve-final2 at e2762eb0 (all three review passes, the block table read through a slice taken once,
main 8b7e9941 merged); old: f8632013 (main just before KV5-1's merge) for G2; main: serve-main-final2 at 8b7e9941 for G3c. A first run of
this file on bf5e0a59 (serve-final, b727c55e) is in g2-final; the table fix changed the decode path, so every cell reads again. Pinned binaries and harness under ~/goinfer-work/kv5-6.
Steps (python3 run_g2_kv5_6_final.py STEP):
  decode  G2's decode cells: the 1.5B at 128, 2,048, 3,900 and 8,000, gemma3-1b at 128, 2,048 and 3,900 (bench_peer.py, A B B A, one
          quad a cell, G4 as amended: any swap-out voids a cell, three voids in a row stop);
  mc3     G2's MC3 cell: 4 clients each on its own slot (-kv-sessions 4), the 1.5B, 6 turns of 128 greedy tokens, cells old new new
          old old new (bench_w7_plain.py);
  ttft    G2's prefill TTFT on the 1.5B and the 7B at K = 512 and 3,900 (ttft_ab.py, three alternating pairs);
  g3c     G3c (g3c.py).
A step whose output exists is skipped, so a rerun resumes. The 7B's decode cells are the night job run-g2-kv5-6-7b.sh."""
import os, subprocess, sys, time

D = os.path.expanduser("~/goinfer-work/kv5-6")
NEW, OLD, MAIN = f"{D}/serve-final2", os.path.expanduser("~/goinfer-work/kv5-2-g2/serve-old"), f"{D}/serve-main-final2"
TREE = f"{D}/new-e2762eb0"
SCRIPTS, KV5 = f"{TREE}/scripts", f"{TREE}/docs/measurements/kv5-metal-2026-11"
OUT = f"{D}/g2-final2"


def say(s):
    print(time.strftime("%H:%M:%S ") + s, flush=True)


def run(name, cmd, env, cwd=SCRIPTS, out_arg=True):
    out = f"{OUT}/{name}.json"
    if os.path.exists(out):
        say(f"== {name}: done before, skipped")
        return
    say(f"== {name}")
    with open(f"{OUT}/{name}.log", "w") as f:
        rc = subprocess.run(cmd + ([out] if out_arg else []), cwd=cwd, env=dict(os.environ, **env), stdout=f, stderr=subprocess.STDOUT).returncode
    say(f"== {name} rc={rc}")
    if rc != 0:
        if os.path.exists(out):
            os.rename(out, out + ".failed")
        sys.exit(75 if any(l.startswith("REFUSED") for l in open(f"{OUT}/{name}.log")) else 1)


def main():
    step = sys.argv[1]
    for f in (NEW, OLD, MAIN, f"{SCRIPTS}/bench_peer.py", f"{SCRIPTS}/bench_w7_plain.py", f"{KV5}/ttft_ab.py", f"{KV5}/g3c.py"):
        if not os.path.exists(f):
            sys.exit(f"missing {f}")
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]
    if "AC Power" not in batt:
        sys.exit(f"not on AC power: {batt}")
    os.makedirs(f"{OUT}/serve-logs", exist_ok=True)
    say(f"{step}: new e2762eb0, old f8632013, main 8b7e9941; {batt}")
    if step == "decode":
        peer = {"BENCH_DEPTH_BACKEND": "metal", "BENCH_ENGINES": "goinfer,goinfer_old", "BENCH_BACKENDS": "metal", "BENCH_REQUIRE_BACKEND": "1",
                "BENCH_SERVE_LOG_DIR": f"{OUT}/serve-logs", "BENCH_LOAD_TIMEOUT": "420", "GOINFER_SERVE_METAL": NEW, "GOINFER_SERVE_METAL_OLD": OLD,
                "BENCH_SWAPOUT_VOID_MB": "0", "BENCH_SWAP_VOID_STOP": "3", "BENCH_IDLE_WAIT": "300", "BENCH_ABBA": "1"}
        bp = ["python3", "-u", "bench_peer.py"]
        run("qwen15", bp, dict(peer, BENCH_MODELS="1.5B", BENCH_DEPTHS="2048,3900"))
        run("deep15", bp, dict(peer, BENCH_MODELS="1.5B", BENCH_PHASES="B", BENCH_DEPTHS="8000", BENCH_DEEP_CTX="8192"))
        run("gemma", bp, dict(peer, BENCH_MODELS="gemma3-1b", BENCH_DEPTHS="2048,3900"))
    elif step == "mc3":
        for i, arm in enumerate(["old", "new", "new", "old", "old", "new"], 1):
            run(f"mc3s4-{i}-{arm}", ["python3", "-u", "bench_w7_plain.py", "--clients", "4", "--engines", "goinfer", "--backend", "metal", "--fixed-nonce",
                                     "--serve-args", "-kv-sessions 4", "--server-log", f"{OUT}/serve-logs/mc3s4-{i}-{arm}.log"],
                {"GOINFER_SERVE_CPU": OLD if arm == "old" else NEW})
    elif step == "ttft":
        for mk in ("1.5B", "7B"):
            run(f"ttft-{mk}", ["python3", "-u", f"{KV5}/ttft_ab.py", OLD, NEW], {"TTFT_MODEL": mk, "TTFT_SCRIPTS": SCRIPTS})
    elif step == "g3c":
        run("g3c", ["python3", "-u", f"{KV5}/g3c.py", MAIN, NEW], {}, cwd=KV5)
    else:
        sys.exit(f"no step {step}")
    say(f"{step}: done")


if __name__ == "__main__":
    main()
