#!/usr/bin/env python3
"""P41's runner and grader (README.md beside this file). Run by run-r13.sh under the timing lock, on the Mac.

TestAttendGroupedHeads_matchesPerHead first (bit-identity and the grouped dispatch counted), then, for each depth, ten
runs of BenchmarkDecodeAtDepth on the 1.5B, each one process whose two sub-benchmarks are the pair (grouped_off, then
grouped_on, sharing nothing but the process); then the 7B control at depth 2048, ten runs. Each run waits for the
1-minute load to settle and records the load and the reclaimable memory at its start; a run started over the gate is void
and re-run at the end.
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile, time

LOAD_MAX = 4.0         # half of the Mac's 8 cores: the void gate
SETTLE = 2.0           # a run starts once the 1-minute load is under this
SETTLE_CAP = 300       # seconds; a run whose wait reaches it is void
FIT_MARGIN = 2 << 30   # the void rule's 2 GB beside the load's own fit
DEPTHS = [128, 2048, 4096, 8192]
CONTROL_DEPTH = 2048
PAIRS = 10
MAX_RERUNS = 6


def settle():
    t0 = time.time()
    while os.getloadavg()[0] >= SETTLE:
        if time.time() - t0 >= SETTLE_CAP:
            return time.time() - t0, False
        time.sleep(2)
    return time.time() - t0, True


def reclaimable(hostmem):
    try:
        return int(subprocess.check_output([hostmem, "-reclaimable"], text=True).strip())
    except Exception:
        return 0


def bench(binary, model, depth):
    env = dict(os.environ, GOINFER_PREQUANT_GGUF=model, GOINFER_BENCH_QUANT="int4", GOINFER_BENCH_DEPTH=str(depth))
    env.pop("GOINFER_ATTN_GROUPED", None)  # the benchmark sets each arm itself
    with tempfile.TemporaryFile("w+") as log:
        rc = subprocess.call([binary, "-test.run", "^$", "-test.bench", "^BenchmarkDecodeAtDepth$", "-test.benchtime", "30x",
                              "-test.count", "1"], env=env, stdout=log, stderr=subprocess.STDOUT)
        log.seek(0)
        text = log.read()
    arms = {}
    for m in re.finditer(r"BenchmarkDecodeAtDepth/(grouped_off|grouped_on)\S*\s+\d+\s+.*?([\d.]+) tok/s", text):
        arms[m.group(1)] = float(m.group(2))
    return rc, arms, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--hostmem", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="Qwen2.5-1.5B-Instruct Q4_K_M .gguf")
    ap.add_argument("--control", required=True, help="Qwen2.5-Coder-0.5B-Instruct Q4_K_M .gguf (ratio 7: both arms per-head)")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--control-only", action="store_true", help="run only the control; grade it with --prior's runs")
    ap.add_argument("--prior", help="raw.jsonl of the earlier sweep whose clean runs are graded with this control")
    a = ap.parse_args()
    if a.control_only and not a.prior:
        sys.exit("--control-only needs --prior")
    raw = open(os.path.join(a.out, "raw.jsonl"), "w")
    grade = os.path.join(a.out, "grade.txt")

    text = subprocess.run([a.bin, "-test.run", "^TestAttendGroupedHeads_matchesPerHead$", "-test.v"],
                          capture_output=True, text=True).stdout
    raw.write(json.dumps({"kind": "identity", "commit": a.commit, "log": text[-2000:]}) + "\n")
    if "--- PASS: TestAttendGroupedHeads_matchesPerHead" not in text:
        open(grade, "w").write("VOID: TestAttendGroupedHeads_matchesPerHead did not pass\n" + text[-2000:])
        sys.exit(1)

    fits = {a.model: 3 << 30, a.control: 2 << 30}  # each model's int4 load, rounded up
    queue = [(a.model, d, i) for d in DEPTHS for i in range(PAIRS)] + [(a.control, CONTROL_DEPTH, i) for i in range(PAIRS)]
    if a.control_only:  # addendum 2026-11-10: the 2026-11-09 night run timed out before its control
        queue = [(a.control, CONTROL_DEPTH, i) for i in range(PAIRS)]
    runs, reruns = [], 0
    while queue:
        model, depth, i = queue.pop(0)
        waited, settled = settle()
        load1, avail = os.getloadavg()[0], reclaimable(a.hostmem)
        void = not settled or load1 > LOAD_MAX or avail < fits[model] + FIT_MARGIN
        start = time.strftime("%H:%M:%S")
        rc, arms, text = bench(a.bin, model, depth)
        if rc != 0 or set(arms) != {"grouped_off", "grouped_on"}:
            void = True
        r = {"kind": "run", "model": os.path.basename(model), "depth": depth, "pair": i, "start": start, "load1": load1,
             "reclaimable": avail, "settle_s": round(waited, 1), "void": void, "rc": rc, "arms": arms}
        if void and rc != 0:
            r["log"] = text[-2000:]
        raw.write(json.dumps(r) + "\n")
        raw.flush()
        print(f"{r['model']} depth {depth} pair {i}: {arms} load {load1:.2f}{' VOID' if void else ''}", flush=True)
        if void:
            if reruns >= MAX_RERUNS:
                open(grade, "w").write(f"VOID: more than {MAX_RERUNS} void runs\n")
                sys.exit(1)
            reruns += 1
            queue.append((model, depth, i))
        else:
            runs.append(r)

    if a.prior:  # the earlier session's clean sweep runs are graded with this session's control
        for line in open(a.prior):
            x = json.loads(line)
            if x.get("kind") == "run" and not x.get("void") and x["model"] == os.path.basename(a.model):
                runs.append(x)

    def ratios(model, depth):
        return [x["arms"]["grouped_on"] / x["arms"]["grouped_off"] for x in runs
                if x["model"] == os.path.basename(model) and x["depth"] == depth]

    lines = [f"P41 R13 sweep at {a.commit}; model {a.model}; control {a.control}"]
    if a.prior:
        lines.append(f"sweep runs from {a.prior} (a separate session); control from this one")
    ctrl = ratios(a.control, CONTROL_DEPTH)
    cmed = statistics.median(ctrl)
    lines.append(f"control ({os.path.basename(a.control)}, depth {CONTROL_DEPTH}): median on/off {cmed:.4f}")
    if abs(cmed - 1.0) > 0.02:
        lines.append("VERDICT: VOID (the control is outside 1.00 +/- 0.02: the fixed arm order is biased)")
        open(grade, "w").write("\n".join(lines) + "\n")
        return
    med, slower, faster = {}, {}, {}
    for d in DEPTHS:
        rs = ratios(a.model, d)
        med[d], slower[d], faster[d] = statistics.median(rs), sum(x < 1 for x in rs), sum(x > 1 for x in rs)
        lines.append(f"depth {d}: median on/off {med[d]:.4f}, {faster[d]} of {len(rs)} pairs faster; "
                     + ", ".join(f"{x:.3f}" for x in rs))
    deep = DEPTHS[-1]
    if all(med[d] >= 0.98 for d in DEPTHS) and med[deep] >= 1.05 and faster[deep] >= 9:
        verdict = "KEEP the default (>= 0.98 at every depth, >= 1.05 at 8192 with 9 of 10 pairs faster)"
    elif med[deep] < 0.98 and slower[deep] >= 8:
        verdict = "TURN IT OFF on arm64 (loses at 8192: median < 0.98, 8 of 10 pairs slower)"
    elif any(med[d] < 0.98 and slower[d] >= 8 for d in DEPTHS[:-1]) and med[deep] >= 1.05 and faster[deep] >= 9:
        keep = [d for d in DEPTHS if med[d] >= 0.98]
        verdict = f"RAISE the floor: loses shallow, keeps at 8192; attnGroupedMinKeys to {min(keep)} (Francis's call)"
    else:
        verdict = "AMBIGUOUS -> parked, to Francis with the numbers"
    lines.append(f"void runs re-run: {reruns}")
    lines.append("VERDICT: " + verdict)
    open(grade, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
