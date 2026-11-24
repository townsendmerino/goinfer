#!/usr/bin/env python3
"""The multi-chain QK A/B (README beside this file): BenchmarkDecodeAtDepth/grouped_on on the CPU with GOINFER_BENCH_FILL=1 and f32 KV,
A = the decoder test binary built on aikit's release (--bin-a), B = the one built on the change (--bin-b), one process a cell, A B B A x 5
blocks (10 pairs) per model and depth. Run under the exclusive timing lock:
  python3 scripts/timing_lock.py run --label cpu-qk-ab -- python3 run_ab.py --bin-a A.test --bin-b B.test --out DIR
Each cell waits for a 1-minute load under 1.0, records its start state, its swap before and after and the process's peak RSS, and is VOID
(re-run in place, listed) if it starts above load 8.0 or swap grows during it. Appends to DIR/cells.tsv; a re-run skips the cells it
already holds a reading for."""
import argparse, datetime, os, re, subprocess, sys, time

MODELS = {"Q05": ("~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "int4"),
          "P3M": ("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf", "q4k")}
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def meminfo(key):
    return next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith(key + ":"))


def swap_mib():
    return meminfo("SwapTotal") - meminfo("SwapFree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-a", required=True)
    ap.add_argument("--bin-b", required=True)
    ap.add_argument("--kv-a", default="", help="GOINFER_BENCH_KV for arm A (default auto's f32)")
    ap.add_argument("--pad-a", action="store_true", help="arm A builds its cache with the row pad on")
    ap.add_argument("--widen-a", action="store_true", help="arm A reads an f16 cache through the widen path (GOINFER_BENCH_F16WIDEN=1)")
    ap.add_argument("--pad-b", action="store_true", help="arm B builds its cache with the row-pad prototype (GOINFER_BENCH_KVPAD=1)")
    ap.add_argument("--kv-b", default="", help="GOINFER_BENCH_KV for arm B (A stays at auto's f32); the f16-read A/B passes f16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="Q05,P3M")
    ap.add_argument("--depths", default="128,3900")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    tsv = os.path.join(a.out, "cells.tsv")
    new = not os.path.exists(tsv)
    done = set()  # resume: a cell with a recorded reading is not re-run
    if not new:
        for line in open(tsv).read().splitlines()[1:]:
            f = line.split("\t")
            if len(f) > 12 and re.fullmatch(r"[0-9.]+", f[12]):
                done.add((f[1], int(f[2]), int(f[3]), int(f[4])))
    with open(tsv, "a") as out:
        if new:
            out.write("start\tmodel\tdepth\tblock\tcell\tarm\tsettle_s\tload1\tmemavail_mib\tswap_before\tswap_after\tmax_rss_kb\ttok_s\twall_s\n")
        for model in a.models.split(","):
            path, quant = MODELS[model]
            for depth in [int(d) for d in a.depths.split(",")]:
                for block in range(1, a.blocks + 1):
                    for cell, arm in enumerate("ABBA", 1):
                        if (model, depth, block, cell) in done:
                            continue
                        for attempt in range(1, 4):
                            if datetime.datetime.now().strftime("%H%M") >= "2125":
                                print("STOP: 21:25 PT reached", flush=True)
                                sys.exit(3)
                            t0 = time.time()
                            while load1() >= 1.0 and time.time() - t0 < 600:
                                time.sleep(10)
                            waited, l1, avail, sw0 = round(time.time() - t0), load1(), meminfo("MemAvailable"), swap_mib()
                            start = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
                            log = os.path.join(a.out, "logs", f"{model}-d{depth}-b{block}-c{cell}-{arm}-a{attempt}.log")
                            env = dict(os.environ, GOINFER_PREQUANT_GGUF=os.path.expanduser(path), GOINFER_BENCH_QUANT=quant,
                                       GOINFER_BENCH_BACKEND="cpu", GOINFER_BENCH_DEPTH=str(depth), GOINFER_BENCH_FILL="1",
                                       GOINFER_BENCH_KV=a.kv_b if arm == "B" else a.kv_a,
                                       GOINFER_BENCH_KVPAD="1" if ((arm == "B" and a.pad_b) or (arm == "A" and a.pad_a)) else "",
                                       GOINFER_BENCH_F16WIDEN="1" if (arm == "A" and a.widen_a) else "")
                            t1 = time.time()
                            with open(log, "w") as lf:
                                rc = subprocess.run(["/usr/bin/time", "-v", a.bin_b if arm == "B" else a.bin_a, "-test.run", "^$", "-test.bench", "^BenchmarkDecodeAtDepth$/^grouped_on$",
                                                     "-test.benchtime", f"{a.tokens}x", "-test.count", "1", "-test.v", "-test.timeout", "20m"],
                                                    env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=os.path.join(REPO, "decoder")).returncode
                            wall, sw1 = time.time() - t1, swap_mib()
                            text = open(log).read()
                            m = re.search(r"grouped_on\S*\s+\d+\s+\d+ ns/op\s+([0-9.]+) tok/s", text) or re.search(r"([0-9.]+) tok/s", text)
                            pad = re.search(r"([0-9.]+) rowpad", text)
                            want_pad = (arm == "B" and a.pad_b and not a.kv_b) or (arm == "A" and a.pad_a and not a.kv_a)
                            if m and rc == 0 and model == "P3M" and want_pad and (not pad or float(pad.group(1)) == 0):
                                sys.exit(f"FAILED {model} d{depth} b{block} c{cell}: arm {arm} asked for the pad and the cache reports rowpad {pad.group(1) if pad else 'missing'}")
                            inpl = re.search(r"([0-9.]+) f16inplace", text)
                            if m and rc == 0 and a.widen_a and (not inpl or (arm == "A") != (float(inpl.group(1)) == 0)):
                                sys.exit(f"FAILED {model} d{depth} b{block} c{cell}: arm {arm}'s f16 in-place read count is {inpl.group(1) if inpl else 'missing'} (want 0 on the widen arm, > 0 on the in-place arm)")
                            rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
                            tok = m.group(1) if (m and rc == 0) else f"FAILED rc={rc}"
                            if "SKIP" in text and not m:
                                tok = "FAILED skipped"
                            if l1 > 8.0:
                                tok = f"VOID start load {l1}"
                            elif sw1 > sw0:
                                tok = f"VOID swap grew {sw0}->{sw1} MiB"
                            out.write(f"{start}\t{model}\t{depth}\t{block}\t{cell}\t{arm}\t{waited}\t{l1}\t{avail}\t{sw0}\t{sw1}\t"
                                      f"{rss.group(1) if rss else ''}\t{tok}\t{wall:.1f}\n")
                            out.flush()
                            print(f"{start} {model} d{depth} b{block} c{cell} {arm} a{attempt} settle {waited}s load {l1} -> {tok} ({wall:.0f}s)", flush=True)
                            if tok.startswith("FAILED"):
                                sys.exit(f"FAILED {model} d{depth} b{block} c{cell}: {tok}")
                            if not tok.startswith("VOID"):
                                break
                        else:
                            sys.exit(f"STOPPED: {model} d{depth} b{block} c{cell} void three times")
    print("CPU-KV-SPEED-DONE", flush=True)


if __name__ == "__main__":
    main()
