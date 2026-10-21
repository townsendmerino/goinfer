#!/usr/bin/env python3
"""B41 gate 2, the real-life cell (README.md here): with Francis's usual apps open and the new reading in the guard, load the 9B
resident on Metal, serve one gate-2(b) pair (opencode #1: turn 1 with its first token forbidden, then turn 2), unload. Samples every
5 s the new reading, the old one, system swap, the memory-pressure level and the server's own memory (task_info TASK_VM_INFO).

  python3 gate2.py --serve <serve built at the B41 branch> --hostmem <cmd/hostmem> --model NAME=GIW --key K --bodies D --ids F --out DIR

PASS (amendment A1): the server's own compressed bytes (task_info TASK_VM_INFO) stay 0, system swap grows by under 4,096 MB, and the pressure level never reaches 4 (critical).
A load the guard refuses is recorded as the reading's verdict at that moment (exit 3), not a fail. Exit 0 PASS, 1 FAIL, 2 VOID.
"""
import argparse, glob, json, os, re, signal, subprocess, sys, threading, time, urllib.request

PORT, GIB, MB = 18213, 1 << 30, 1 << 20
SESSION = "pilot-qwen35-9b-20k-opencode-chain-r1"


def sh(*a):
    return subprocess.run(list(a), capture_output=True, text=True).stdout


def old_bytes():
    o = sh("vm_stat")
    ps = int(re.search(r"page size of (\d+) bytes", o).group(1))
    return sum(int(re.search(re.escape(k) + r"\s+(\d+)\.", o).group(1))
               for k in ("Pages free:", "Pages inactive:", "Pages speculative:", "Pages purgeable:")) * ps


def swap_used():
    m = re.search(r"used = ([\d.]+)M", sh("sysctl", "-n", "vm.swapusage"))
    return float(m.group(1)) * MB if m else 0.0


def own_vm(hostmem, pid):
    """The process's own memory from task_info(TASK_VM_INFO) through `hostmem -pid` (amendment A1): resident, internal, compressed
    (in the compressor, in memory or swapped; macOS keeps no separate per-process disk-swap figure), footprint. None when unreadable."""
    out = sh(hostmem, "-pid", str(pid)).split()
    return {out[i]: int(out[i + 1]) for i in range(0, len(out) - 1, 2)} if out else None


class Sampler(threading.Thread):
    def __init__(self, hostmem, log):
        super().__init__(daemon=True)
        self.hostmem, self.log, self.pid, self.phase, self.rows, self.stop = hostmem, log, None, "before", [], False

    def run(self):
        while not self.stop:
            r = {"t": time.strftime("%H:%M:%S"), "phase": self.phase, "new": int(sh(self.hostmem).strip() or 0), "old": old_bytes(),
                 "swap": swap_used(), "pressure": int(sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip() or 0),
                 "own": own_vm(self.hostmem, self.pid) if self.pid else None}
            self.rows.append(r)
            o = r["own"]
            own = "-" if not o else f"resident {o['resident'] / GIB:.2f} GiB, compressed {o['compressed'] / MB:.1f} MB"
            print(f"{r['t']} {r['phase']:8s} new {r['new'] / GIB:.2f} GiB  old {r['old'] / GIB:.2f} GiB  swap {r['swap'] / GIB:.2f} GiB  "
                  f"pressure {r['pressure']}  own {own}",
                  file=self.log, flush=True)
            time.sleep(5)


def send(rec, ban=None):
    body = json.loads(rec["body"]) if isinstance(rec["body"], str) else dict(rec["body"])
    body.update(stream=False, temperature=0, max_tokens=24)
    body.pop("stream_options", None)
    body.pop("logit_bias", None)
    if ban is not None:
        body["logit_bias"] = {str(ban): -100}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}" + rec["path"], data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    urllib.request.urlopen(req, timeout=1800).read()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    for k in ("serve", "hostmem", "model", "key", "bodies", "ids", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = Sampler(a.hostmem, sys.stdout)
    s.start()
    time.sleep(16)  # three readings before the load
    fs = sorted(glob.glob(os.path.join(a.bodies, SESSION, "*.json")))
    t1rec, t2rec = json.load(open(fs[1])), json.load(open(fs[2]))
    ids = {os.path.realpath(r["file"]): r["ids"] for r in map(json.loads, open(a.ids)) if r.get("ids")}
    i1, i2 = ids.get(os.path.realpath(fs[1])), ids.get(os.path.realpath(fs[2]))
    ban = i2[len(i1)] if i1 and i2 and i2[:len(i1)] == i1 and len(i2) > len(i1) else None
    log = open(os.path.join(a.out, "serve.log"), "w")
    s.phase = "load"
    p = subprocess.Popen([a.serve, "--model", a.model, "--backend", "metal", "--ctx", "20480", "--kv-sessions", "1", "--ids-key-file", a.key,
                          "--addr", f"127.0.0.1:{PORT}", "--log-requests"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    s.pid = p.pid
    rc = 0
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1).read()
                break
            except Exception:
                if p.poll() is not None:
                    print("VOID: the server exited during load")
                    return 2
                time.sleep(0.5)
        log.flush()
        path = [ln.strip() for ln in open(log.name, errors="replace") if "decode path:" in ln or "BuildResident declined" in ln]
        print("load:", *path, sep="\n  ")
        if not any("metal-resident" in x for x in path):
            print("VERDICT REFUSED: the guard kept the 9B off Metal at this reading (recorded; not a fail)")
            rc = 3
        else:
            s.phase = "serve"
            print(f"turn 1 (ban {ban}): {send(t1rec, ban):.1f} s; turn 2: {send(t2rec):.1f} s")
    finally:
        s.phase = "unload"
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=60)
        s.pid = None
        s.phase = "after"
        time.sleep(16)
        s.stop = True
        s.join(timeout=10)
        json.dump(s.rows, open(os.path.join(a.out, "samples.json"), "w"), indent=1)
    if rc:
        return rc
    base = [r for r in s.rows if r["phase"] == "before"]
    own = [r["own"]["compressed"] for r in s.rows if r["own"]]
    swap_growth = max(r["swap"] for r in s.rows) - min(r["swap"] for r in base)
    pmax = max(r["pressure"] for r in s.rows)
    ok_own, ok_swap, ok_p = bool(own) and max(own) == 0, swap_growth < 4096 * MB, pmax < 4
    print(f"own compressed max {max(own) / MB if own else float('nan'):.1f} MB over {len(own)} samples (bar 0); system swap growth "
          f"{swap_growth / MB:.0f} MB (bar < 4096); pressure max {pmax} (bar < 4)")
    if not own:
        print("VERDICT VOID (no own-swap sample while the server ran)")
        return 2
    print(f"VERDICT {'PASS' if ok_own and ok_swap and ok_p else 'FAIL'}")
    return 0 if ok_own and ok_swap and ok_p else 1


if __name__ == "__main__":
    sys.exit(main())
