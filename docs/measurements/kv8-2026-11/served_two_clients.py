#!/usr/bin/env python3
"""KV8's served reading (README.md here): two clients taking turns on one server, one KV slot against two, A B B A in one session.

  python3 served_two_clients.py --serve <bin> --model <path> --out <dir> [--fit-only]

Prints a VERDICT line and writes <out>/turns.jsonl. Exit 0 PASS, 1 FAIL, 2 VOID, 3 PARKED.
"""
import argparse, json, os, statistics, subprocess, sys, threading, time, urllib.request

PORT, CTX, MAX_NEW, ROUNDS = 18219, 8192, 24, 3
BASE = f"http://127.0.0.1:{PORT}"
PASS_BAR, FAIL_BAR = 0.5, 0.7
SWAP_MB = 512
LOAD_VOID = 8.0


def preamble(name):
    """About 2,000 tokens of plain text no two clients share."""
    return "".join(f"Note {i} of client {name}: the {name} ledger entry {i * 7 % 101} moved to shelf {i * 13 % 97} on day {i % 28 + 1}.\n" for i in range(70))


def readings():
    load = os.getloadavg()[0]
    avail = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    vram = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    return load, avail, int(vram)


class Server:
    def __init__(self, a, slots, log):
        self.log = open(log, "w")
        argv = [a.serve, "--model", a.model, "--backend", "cuda", "--ctx", str(CTX), "--addr", f"127.0.0.1:{PORT}", "--kv-sessions", str(slots), "--log-requests"]
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.peak_swap, self.peak_hwm, self.stop = 0, 0, threading.Event()
        threading.Thread(target=self.sample, daemon=True).start()
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited {self.proc.returncode} (see {log})")
            if time.time() - t0 > 300:
                raise RuntimeError("the server did not listen in 300 s")
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except OSError:
                time.sleep(0.5)
        self.load_s = time.time() - t0

    def sample(self):
        while not self.stop.wait(1.0):
            try:
                for l in open(f"/proc/{self.proc.pid}/status"):
                    if l.startswith("VmSwap"):
                        self.peak_swap = max(self.peak_swap, int(l.split()[1]) // 1024)
                    elif l.startswith("VmHWM"):
                        self.peak_hwm = max(self.peak_hwm, int(l.split()[1]) // 1024)
            except OSError:
                return

    def close(self):
        self.stop.set()
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()


def model_id():
    return json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=5))["data"][0]["id"]


def turn(prompt):
    """One request. It returns the reply's text, the time to its first chunk, and usage."""
    body = {"model": model_id(), "prompt": prompt, "max_tokens": MAX_NEW, "temperature": 0, "seed": 1, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, text, usage = time.time(), None, [], {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                if c.get("text"):
                    first = first or time.time()
                    text.append(c["text"])
    return "".join(text), (first or time.time()) - t0, usage


def cell(a, label, slots, pair, out):
    load, avail, vram0 = readings()
    rec = {"cell": label, "slots": slots, "pair": pair, "load": load, "avail_mb": avail, "vram_before_mib": vram0, "void": load > LOAD_VOID}
    print(f"[{time.strftime('%H:%M:%S')}] cell {label}: {slots} slot(s), load {load:.2f}, {avail} MB available, {vram0} MiB on the card", flush=True)
    if rec["void"]:
        print(f"VOID CELL {label}: load {load:.2f} over {LOAD_VOID}", flush=True)
        return rec, []
    s = Server(a, slots, os.path.join(out, f"serve-{label}.log"))
    rec["load_s"], rec["vram_loaded_mib"] = s.load_s, readings()[2]
    prompts = {"A": preamble("amber"), "B": preamble("birch")}
    turns = []
    try:
        for r in range(ROUNDS):
            for name in "AB":
                text, ttft, u = turn(prompts[name])
                turns.append({"cell": label, "slots": slots, "round": r, "client": name, "ttft_s": ttft, "reused": u.get("prefill_reused_tokens"), "prompt_tokens": u.get("prompt_tokens"), "reply": text})
                print(f"[{time.strftime('%H:%M:%S')}]   {label} round {r} {name}: {u.get('prompt_tokens')} tokens, reused {u.get('prefill_reused_tokens')}, first token {ttft:.2f} s", flush=True)
                prompts[name] += text + f"\nQuestion {r}: which shelf next?\nAnswer:"
        # Both clients at once: the replies must still be each conversation's own.
        got = {}
        ts = [threading.Thread(target=lambda n=n: got.__setitem__(n, turn(prompts[n]))) for n in "AB"]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for name in "AB":
            text, ttft, u = got[name]
            turns.append({"cell": label, "slots": slots, "round": "both", "client": name, "ttft_s": ttft, "reused": u.get("prefill_reused_tokens"), "prompt_tokens": u.get("prompt_tokens"), "reply": text})
            print(f"[{time.strftime('%H:%M:%S')}]   {label} both at once {name}: reused {u.get('prefill_reused_tokens')}, first token {ttft:.2f} s", flush=True)
    finally:
        rec["peak_swap_mb"], rec["peak_hwm_mb"] = s.peak_swap, s.peak_hwm
        s.close()
    print(f"[{time.strftime('%H:%M:%S')}]   {label}: loaded in {rec['load_s']:.1f} s, {rec['vram_loaded_mib']} MiB on the card, peak VmSwap {s.peak_swap} MB, VmHWM {s.peak_hwm} MB", flush=True)
    return rec, turns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fit-only", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.fit_only:
        for slots in (1, 2):
            log = os.path.join(a.out, f"fit-{slots}.log")
            s = Server(a, slots, log)
            print(f"{slots} slot(s): loaded in {s.load_s:.1f} s, {readings()[2]} MiB on the card")
            s.close()
            for l in open(log):
                if any(k in l for k in ("fit:", "KV plan", "recurrent checkpoint", "KV slots", "resident context", "decline")):
                    print("  " + l.rstrip()[:200])
            time.sleep(3)
        return 0
    cells, turns = [], []
    done = {}
    for label, slots, pair in (("A1", 1, 1), ("B1", 2, 1), ("B2", 2, 2), ("A2", 1, 2)):
        rec, ts = cell(a, label, slots, pair, a.out)
        if rec["void"]:  # re-run in the same order, once, and the record lists it
            cells.append(rec)
            time.sleep(60)
            rec, ts = cell(a, label + "r", slots, pair, a.out)
        cells.append(rec)
        turns += ts
        done[label] = ts
        time.sleep(3)
    with open(os.path.join(a.out, "turns.jsonl"), "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
        for t in turns:
            f.write(json.dumps(t) + "\n")
    if any(c["void"] for c in cells if c["cell"].endswith("r")):
        print("VERDICT: VOID (a re-run cell was also void)")
        return 2

    def follow(ts):
        return statistics.median(t["ttft_s"] for t in ts if t["round"] in (1, 2))

    ratios = []
    for pair in (1, 2):
        one = next(ts for l, ts in done.items() if l.startswith("A") and l[1] == str(pair))
        two = next(ts for l, ts in done.items() if l.startswith("B") and l[1] == str(pair))
        ratios.append(follow(two) / follow(one))
        print(f"pair {pair}: follow-up median first token, one slot {follow(one):.2f} s, two slots {follow(two):.2f} s, ratio {ratios[-1]:.3f}")
    two = [t for l, ts in done.items() if l.startswith("B") for t in ts]
    one = [t for l, ts in done.items() if l.startswith("A") for t in ts]
    unreused = [t for t in two if t["round"] in (1, 2) and not t["reused"]]
    print(f"two slots: {len(unreused)} of {sum(t['round'] in (1, 2) for t in two)} follow-ups reused nothing")
    # A conversation's replies against the one-slot arm's, turn by turn, up to its first difference: past that the prompts differ.
    differ = 0
    for pair in (1, 2):
        for name in "AB":
            for r in (0, 1, 2, "both"):
                x = next(t for t in done[f"A{pair}"] if t["client"] == name and t["round"] == r)
                y = next(t for t in done[f"B{pair}"] if t["client"] == name and t["round"] == r)
                if x["reply"] != y["reply"]:
                    differ += 1
                    print(f"pair {pair} client {name} round {r}: the replies differ: one slot {x['reply']!r}, two slots {y['reply']!r}")
                    break
    print(f"{differ} of 4 conversations' replies differ between the arms")
    swap = max(c.get("peak_swap_mb", 0) for c in cells)
    print(f"peak VmSwap {swap} MB (the bar: {SWAP_MB})")
    worst = max(ratios)
    if swap > SWAP_MB or differ or unreused or worst > FAIL_BAR:
        print(f"VERDICT: FAIL (ratio {worst:.3f}, {differ} differ, {len(unreused)} unreused, swap {swap} MB)")
        return 1
    if worst > PASS_BAR:
        print(f"VERDICT: PARKED (ratio {worst:.3f} in the band {PASS_BAR}-{FAIL_BAR})")
        return 3
    print(f"VERDICT: PASS (worst ratio {worst:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
