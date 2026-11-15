#!/usr/bin/env python3
"""KV8's replay of KV0-H's recorded agent requests (README.md here, the third addendum): serve before the message-boundary checkpoint against serve after it, A B B A,
one server per cell, every recorded session in order through the one slot, each request replayed as recorded (its own route and body) on a small hybrid.

  python3 replay_kv0h.py --main <bin> --change <bin> --model <path> --bodies <kv0h bodies parent dir> --out <dir> [--ctx N] [--max-new N]

--bodies holds bodies-study-h20/<session>/NNNN.json and check-cc-rule/bodies/<session>/NNNN.json (KV0-H's, outside the repository). Writes <out>/requests.jsonl.
Exit 0 PASS, 1 FAIL, 2 VOID, 3 PARKED.
"""
import argparse, glob, json, os, re, statistics, subprocess, sys, threading, time, urllib.request

PORT = 18219
BASE = f"http://127.0.0.1:{PORT}"
LOAD_VOID, SWAP_MB = 8.0, 512
RATIO_PASS, RATIO_FAIL = 1.00, 1.10


def readings():
    load = os.getloadavg()[0]
    avail = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    return load, avail


def sessions(bodies):
    out = []
    for kind in ("bodies-study-h20", "check-cc-rule/bodies"):
        for d in sorted(glob.glob(os.path.join(bodies, kind, "*"))):
            fs = sorted(glob.glob(os.path.join(d, "*.json")))
            if fs:
                out.append((os.path.basename(d), fs))
    return out


class Server:
    def __init__(self, binary, model, ctx, log, ids):
        self.log = open(log, "w")
        argv = [binary, "--model", model, "--backend", "cuda", "--ctx", str(ctx), "--addr", f"127.0.0.1:{PORT}", "--kv-sessions", "1", "--log-requests", "--log-token-ids", ids]
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.peak_swap, self.stop = 0, threading.Event()
        threading.Thread(target=self.sample, daemon=True).start()
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited {self.proc.returncode} (see {log})")
            if time.time() - t0 > 300:
                raise RuntimeError("the server did not listen in 300 s")
            try:
                self.model = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=1))["data"][0]["id"]
                break
            except OSError:
                time.sleep(0.5)

    def sample(self):
        while not self.stop.wait(1.0):
            try:
                for l in open(f"/proc/{self.proc.pid}/status"):
                    if l.startswith("VmSwap"):
                        self.peak_swap = max(self.peak_swap, int(l.split()[1]) // 1024)
            except OSError:
                return

    def close(self):
        self.stop.set()
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()


def replay(model, rec, max_new):
    """One recorded request, as recorded except the model name, greedy sampling and a short reply. It reads the stream to its end."""
    body = rec["body"] if isinstance(rec["body"], dict) else json.loads(rec["body"])
    body["model"], body["temperature"], body["stream"] = model, 0, True
    for k in ("max_tokens", "max_output_tokens"):
        if k in body:
            body[k] = max_new
    if rec["path"].startswith("/v1/responses"):
        body["max_output_tokens"] = max_new
    req = urllib.request.Request(BASE + rec["path"], json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for _ in r:
            pass


def parse(log, ids):
    """Per request, in order: the request line's fields, and the snapshots the server took and restored while serving it."""
    reqs, took, restored = [], [], 0
    for l in open(log):
        if l.startswith("kv snapshot: took "):
            m = re.match(r"kv snapshot: took (\d+) rows \((\d+) MiB host\) from slot \d+ in ([\d.]+) ms", l)
            took.append({"rows": int(m[1]), "mib": int(m[2]), "ms": float(m[3])})
        elif l.startswith("kv snapshot: restored "):
            restored = int(re.match(r"kv snapshot: restored (\d+) rows", l)[1])
        elif l.startswith("request: "):
            f = dict(re.findall(r"(\w+)=(\S+)", l))
            reqs.append({"prompt_tokens": int(f["prompt_tokens"]), "ttft_s": int(f["ttft"].rstrip("ms")) / 1000, "reused": int(f["reused"]),
                         "snap": int(f.get("snap", 0)), "ckpt": int(f.get("ckpt_restore", 0)), "restored_rows": restored, "took": took})
            took, restored = [], 0
    gens = [json.loads(l)["gen"] for l in open(ids)]
    assert len(gens) == len(reqs), (len(gens), len(reqs))
    for r, g in zip(reqs, gens):
        r["gen"] = g
    return reqs


def cell(label, which, binary, a, plan):
    load, avail = readings()
    rec = {"cell": label, "build": which, "load": load, "avail_mb": avail, "void": load > LOAD_VOID}
    print(f"[{time.strftime('%H:%M:%S')}] cell {label} ({which}): load {load:.2f}, {avail} MB available", flush=True)
    if rec["void"]:
        print(f"VOID CELL {label}: load {load:.2f} over {LOAD_VOID}", flush=True)
        return rec, []
    log, ids = os.path.join(a.out, f"serve-{label}.log"), os.path.join(a.out, f"tokenids-{label}.jsonl")
    s = Server(binary, a.model, a.ctx, log, ids)
    order = []
    try:
        for name, files in plan:
            for i, f in enumerate(files):
                replay(s.model, json.load(open(f)), a.max_new)
                order.append((name, i))
            print(f"[{time.strftime('%H:%M:%S')}]   {label} {name}: {len(files)} requests", flush=True)
    finally:
        rec["peak_swap_mb"] = s.peak_swap
        s.close()
    reqs = parse(log, ids)
    assert len(reqs) == len(order), (len(reqs), len(order))
    for r, (name, i) in zip(reqs, order):
        r.update(cell=label, build=which, session=name, idx=i)
    return rec, reqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--bodies", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ctx", type=int, default=24576)
    p.add_argument("--max-new", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plan = sessions(a.bodies)
    print(f"{len(plan)} sessions, {sum(len(f) for _, f in plan)} requests per cell", flush=True)
    cells, done = [], {}
    for label, which, binary in [("A1", "main", a.main), ("B1", "change", a.change), ("B2", "change", a.change), ("A2", "main", a.main)]:
        rec, rs = cell(label, which, binary, a, plan)
        if rec["void"]:
            cells.append(rec)
            time.sleep(60)
            rec, rs = cell(label + "r", which, binary, a, plan)
        cells.append(rec)
        done[label] = rs
        time.sleep(3)
    with open(os.path.join(a.out, "requests.jsonl"), "w") as f:
        for c in cells:
            f.write(json.dumps({"cell_record": c}) + "\n")
        for rs in done.values():
            for r in rs:
                f.write(json.dumps(r) + "\n")
    if any(c["void"] for c in cells if c["cell"].endswith("r")):
        print("VERDICT: VOID (a re-run cell was also void)")
        return 2
    verdict, notes = "PASS", []

    def worse(v):
        nonlocal verdict
        verdict = max(verdict, v, key=["PASS", "PARKED", "FAIL"].index)

    def kind(r, prev):
        if r["snap"]:
            for q in prev:
                if len(q["took"]) >= 2 and q["took"][0]["rows"] < q["took"][1]["rows"] and q["took"][0]["rows"] == r["snap"]:
                    return "boundary"
            return "keep/other snapshot"
        if r["ckpt"]:
            return "end-of-prompt checkpoint"
        return "live state" if r["reused"] else "cold"

    for pair, (old, new) in {"pair 1": ("A1", "B1"), "pair 2": ("A2", "B2")}.items():
        o, n = done[old], done[new]
        ratios, tot_o, tot_n, kinds_o, kinds_n, fewer, differ = [], 0.0, 0.0, {}, {}, 0, 0
        for i, (ro, rn) in enumerate(zip(o, n)):
            first = ro["idx"] == 0
            if rn["gen"] != ro["gen"]:
                differ += 1
            if rn["reused"] < ro["reused"]:
                fewer += 1
            if first:
                continue
            ratios.append(rn["ttft_s"] / ro["ttft_s"])
            tot_o, tot_n = tot_o + ro["ttft_s"], tot_n + rn["ttft_s"]
            prev_o = [x for x in o[:i] if x["session"] == ro["session"]]
            prev_n = [x for x in n[:i] if x["session"] == rn["session"]]
            ko, kn = kind(ro, prev_o), kind(rn, prev_n)
            kinds_o[ko], kinds_n[kn] = kinds_o.get(ko, 0) + 1, kinds_n.get(kn, 0) + 1
        med, tot = statistics.median(ratios), tot_n / tot_o
        print(f"{pair}: {len(ratios)} follow-ups; first token change over main, median {med:.3f}, total {tot_n:.1f} s over {tot_o:.1f} s = {tot:.3f}")
        print(f"{pair}: follow-ups by what served them, main {dict(sorted(kinds_o.items()))}; change {dict(sorted(kinds_n.items()))}")
        print(f"{pair}: requests with a different reply {differ}; requests reusing fewer rows than main {fewer}")
        if differ:
            notes.append(f"{pair}: {differ} replies differ"); worse("FAIL")
        if fewer:
            notes.append(f"{pair}: {fewer} requests reuse fewer rows"); worse("FAIL")
        for v in (med, tot):
            worse("FAIL" if v > RATIO_FAIL else "PARKED" if v > RATIO_PASS else "PASS")
        if sum(v for k, v in kinds_n.items() if k in ("boundary", "keep/other snapshot")) < sum(v for k, v in kinds_o.items() if k in ("boundary", "keep/other snapshot")):
            notes.append(f"{pair}: fewer snapshot restores than main"); worse("FAIL")
    for c in cells:
        if c.get("peak_swap_mb", 0) > SWAP_MB:
            notes.append(f"{c['cell']}: VmSwap {c['peak_swap_mb']} MB"); worse("FAIL")
    print("VERDICT:", verdict, "; ".join(notes))
    return {"PASS": 0, "FAIL": 1, "PARKED": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
