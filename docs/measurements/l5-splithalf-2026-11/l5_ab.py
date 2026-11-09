#!/usr/bin/env python3
"""L5's A/B runner and grader (README.md beside this file). Run by run-l5.sh under the timing lock.

Fires check first, then the visits ON OFF OFF ON x5, each its own process of the decoder test binary running
BenchmarkDecode -count 4. Each visit records its arm, the load average and MemAvailable at its start, its tok/s
samples and the process's peak VmHWM. A visit started under heavy load is VOID and re-run at the end in its arm.
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile, time

LOAD_MAX = 8.0           # half of nobara's 16 cores
MEM_MIN = 5 * 1024**3    # the ON arm's fit (about 3 GB) plus 2 GB
ORDER = ["on", "off", "off", "on"] * 5
MAX_RERUNS = 4


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def run(binary, args, env):
    """Runs binary, sampling its VmHWM until it exits; returns (output, peak bytes, exit code)."""
    with tempfile.TemporaryFile("w+") as log:
        p = subprocess.Popen([binary] + args, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        peak = 0
        while p.poll() is None:
            try:
                for line in open(f"/proc/{p.pid}/status"):
                    if line.startswith("VmHWM:"):
                        peak = max(peak, int(line.split()[1]) * 1024)
            except OSError:
                pass
            time.sleep(0.2)
        log.seek(0)
        return log.read(), peak, p.returncode


def arm_env(arm):
    env = dict(os.environ)
    env.pop("GOINFER_W4A8_SPLITHALF", None)
    if arm == "on":
        env["GOINFER_W4A8_SPLITHALF"] = "1"
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    raw = open(os.path.join(a.out, "raw.jsonl"), "w")

    text, _, rc = run(a.bin, ["-test.run", "^TestW4A8SplitHalfFires_onBenchModel$", "-test.v"], arm_env("on"))
    m = re.search(r"(\d+) tensors repacked, (\d+) skipped.*?([\d.]+) MiB", text)
    fires = {"kind": "fires", "commit": a.commit, "model": a.model, "rc": rc, "log": text[-2000:]}
    if m:
        fires.update(repacked=int(m.group(1)), skipped=int(m.group(2)), mib=float(m.group(3)))
    raw.write(json.dumps(fires) + "\n")
    raw.flush()
    if rc != 0 or "--- PASS" not in text or not m or int(m.group(1)) == 0:
        open(os.path.join(a.out, "grade.txt"), "w").write("VOID: the split-half repack did not fire\n" + text[-2000:])
        sys.exit(1)

    visits, queue, reruns = [], list(enumerate(ORDER)), 0
    while queue:
        slot, arm = queue.pop(0)
        load1 = os.getloadavg()[0]
        avail = mem_available()
        void = load1 > LOAD_MAX or avail < MEM_MIN
        start = time.strftime("%H:%M:%S")
        text, peak, rc = run(a.bin, ["-test.run", "^$", "-test.bench", "^BenchmarkDecode$", "-test.benchtime", "30x",
                                     "-test.count", "4"], arm_env(arm))
        tps = [float(x) for x in re.findall(r"([\d.]+) tok/s", text)]
        v = {"kind": "visit", "slot": slot, "arm": arm, "start": start, "load1": load1, "mem_available": avail,
             "void": void, "rc": rc, "tok_s": tps, "vmhwm": peak}
        if rc != 0 or len(tps) != 4:
            v["void"] = True
            v["log"] = text[-2000:]
        raw.write(json.dumps(v) + "\n")
        raw.flush()
        print(f"slot {slot} {arm}: {tps} load {load1:.2f} avail {avail / 1e9:.1f} GB{' VOID' if v['void'] else ''}", flush=True)
        if v["void"]:
            if reruns >= MAX_RERUNS:
                open(os.path.join(a.out, "grade.txt"), "w").write(f"VOID: more than {MAX_RERUNS} void visits\n")
                sys.exit(1)
            reruns += 1
            queue.append((slot, arm))
        else:
            visits.append(v)

    by_slot = {v["slot"]: v for v in visits}
    med = {s: statistics.median(by_slot[s]["tok_s"][1:]) for s in by_slot}
    gains = []
    for k in range(0, len(ORDER), 2):  # pairs: (on, off), (off, on), ...
        s1, s2 = k, k + 1
        on, off = (s1, s2) if ORDER[s1] == "on" else (s2, s1)
        gains.append(med[on] / med[off] - 1)
    g = statistics.median(gains)
    pos = sum(x > 0 for x in gains)
    peaks = {arm: max(v["vmhwm"] for v in visits if v["arm"] == arm) for arm in ("on", "off")}
    if g < 0.02:
        verdict = "KILL L5 (median gain under +2.0%)"
    elif pos >= 9:
        verdict = "BUILD split-half-only (median gain at or above +2.0%, at least 9 of 10 pairs positive)"
    else:
        verdict = "AMBIGUOUS -> parked, to Francis (median at or above +2.0%, fewer than 9 positive pairs)"
    lines = [f"L5 A/B at {a.commit}, {a.model}",
             f"repack: {fires['repacked']} tensors, {fires['skipped']} skipped, {fires['mib']:.1f} MiB second copy",
             "pair gains: " + ", ".join(f"{x * 100:+.2f}%" for x in gains),
             f"median gain {g * 100:+.2f}%, {pos} of {len(gains)} pairs positive",
             f"peak VmHWM (recorded, not graded): ON {peaks['on'] / 2**20:.0f} MiB, OFF {peaks['off'] / 2**20:.0f} MiB",
             f"void visits re-run: {reruns}",
             f"VERDICT: {verdict}"]
    open(os.path.join(a.out, "grade.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
