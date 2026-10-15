#!/usr/bin/env python3
"""G40-2 (docs/tasks/task-cpu-prefill-memory-2026-10.md §4): CPU serve's memory through one long cold prompt.

One fresh `serve -backend cpu` per cell (arm x model), at serve's defaults. One request: a cold prompt of about 20,193
tokens (KV9's preamble: this repository's text at e42db592, cut to a character budget), max_tokens 8, temperature 0.
During the request the server's footprint (phys_footprint, `footprint -p`) and RSS are sampled every 250 ms and the
maximum kept; 10 s after it the footprint is read again. Swap-used is read before and after the cell.

PASS needs, on the after arm of each model:
  - peak footprint <= load footprint + the session's KV by geometry + 768 MiB;
  - peak RSS <= the weight file's size + the KV by geometry + 768 MiB (RSS counts the mapped weights);
  - swap-used at the end <= at the start + 32 MiB;
  - the footprint 10 s after the request <= load footprint + KV + 256 MiB.
A peak within 64 MiB of its bound is PARKED (re-read at a 100 ms interval; never re-rolled). A cell is VOID (not a fail)
when less than 1 GiB of swap is free or less than 4 GiB of memory is available at its start, or, for the before arm of
the 1.5B, less than 7 GiB (its own arithmetic says it would swap otherwise).

  python3 g40_mem.py --before <serve> --after <serve> --out <dir> --repo <checkout with e42db592>
"""
import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request

PIN = "e42db592"
SOURCES = ["CLAUDE.md", "docs/tasks/task-kv-architecture-2026-10.md", "docs/tasks/task-forward-architecture-2026-10.md",
           "RELEASING.md", "docs/server.md"]
CHARS = int(20000 * 3.6)  # KV9's budget; serve counted 20,193 tokens
HOME = os.path.expanduser("~")
MODELS = {  # name: (gguf, layers, kv heads, head dim)
    "0.5b": (f"{HOME}/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", 24, 2, 64),
    "1.5b": (f"{HOME}/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 28, 2, 128),
}
MiB = 1 << 20
PAGE = 16384


def preamble(repo):
    text = ""
    for p in SOURCES:
        text += subprocess.run(["git", "-C", repo, "show", f"{PIN}:{p}"], capture_output=True, text=True, check=True).stdout
        if len(text) >= CHARS:
            break
    return text[:CHARS]


def available():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    pages = sum(int(m.group(1)) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
                if (m := re.search(k + r":\s+(\d+)", out)))
    return pages * PAGE


def swap():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)M", out)
    free = re.search(r"free = ([\d.]+)M", out)
    return float(used.group(1)) * MiB if used else 0.0, float(free.group(1)) * MiB if free else 0.0


def footprint(pid):
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"Footprint: ([\d.]+) (KB|MB|GB)", out)
    return float(m.group(1)) * {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[m.group(2)] if m else None


def rss(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else None


def wait_ready(port, proc, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def sidecar(gguf):
    base = gguf[:-len(".gguf")]
    for c in sorted(os.listdir(os.path.dirname(gguf))):
        p = os.path.join(os.path.dirname(gguf), c)
        if p.startswith(base) and "cpu-" in c and c.endswith(".giw"):
            return p
    return gguf


def cell(arm, binary, name, text, port, logs, interval):
    gguf, layers, kvh, hd = MODELS[name]
    rec = {"arm": arm, "model": name, "binary": binary}
    avail, (sw_used, sw_free) = available(), swap()
    floor = 7 << 30 if (arm == "before" and name == "1.5b") else 4 << 30
    if sw_free < (1 << 30) or avail < floor:
        rec.update(verdict="VOID", why=f"available {avail/MiB:.0f} MiB (floor {floor/MiB:.0f}), swap free {sw_free/MiB:.0f} MiB")
        return rec
    log = open(f"{logs}/serve-{arm}-{name}.log", "w")
    proc = subprocess.Popen([binary, "-backend", "cpu", "-model", f"m={gguf}", "-addr", f"127.0.0.1:{port}"],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not wait_ready(port, proc):
            rec.update(verdict="VOID", why="the server did not come up")
            return rec
        time.sleep(3)
        load_fp = footprint(proc.pid)
        peak = {"fp": 0.0, "rss": 0}
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                f, r = footprint(proc.pid), rss(proc.pid)
                if f:
                    peak["fp"] = max(peak["fp"], f)
                if r:
                    peak["rss"] = max(peak["rss"], r)
                stop.wait(interval)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        body = {"model": "m", "max_tokens": 8, "temperature": 0, "messages": [
            {"role": "system", "content": text}, {"role": "user", "content": "In one word, what is this document about?"}]}
        t0 = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            usage = json.loads(r.read()).get("usage", {})
        wall = time.time() - t0
        stop.set()
        th.join()
        time.sleep(10)
        after_fp = footprint(proc.pid)
        sw_end, _ = swap()
        n = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
        kv = n * layers * 2 * kvh * hd * 4
        wfile = os.path.getsize(sidecar(gguf))
        rec.update(prompt_tokens=usage.get("prompt_tokens"), wall_s=round(wall, 1), load_fp=load_fp, peak_fp=peak["fp"],
                   peak_rss=peak["rss"], after_fp=after_fp, kv=kv, weight_file=wfile, swap_grown=sw_end - sw_used,
                   bound_fp=load_fp + kv + 768 * MiB, bound_rss=wfile + kv + 768 * MiB, bound_after=load_fp + kv + 256 * MiB)
        checks = {"peak footprint": rec["peak_fp"] <= rec["bound_fp"], "peak RSS": rec["peak_rss"] <= rec["bound_rss"],
                  "swap": rec["swap_grown"] <= 32 * MiB, "returned": rec["after_fp"] <= rec["bound_after"]}
        near = (0 <= rec["bound_fp"] - rec["peak_fp"] < 64 * MiB) or (0 <= rec["bound_rss"] - rec["peak_rss"] < 64 * MiB)
        rec["checks"] = checks
        rec["verdict"] = "PASS" if all(checks.values()) and not near else ("PARKED" if all(checks.values()) else "FAIL")
        return rec
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()


def warm(binary, port, logs):
    """Starts the server once per model and stops it when it is ready, so each model's CPU .giw exists before any
    measured cell: a cell that builds it would count the build's heap in its load footprint."""
    for name, (gguf, *_rest) in MODELS.items():
        log = open(f"{logs}/warm-{name}.log", "w")
        proc = subprocess.Popen([binary, "-backend", "cpu", "-model", f"m={gguf}", "-addr", f"127.0.0.1:{port}"],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ok = wait_ready(port, proc, limit=1200)
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
        log.close()
        print(json.dumps({"warm": name, "ready": ok, "sidecar": sidecar(gguf)}), flush=True)
        port += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="")
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--arms", default="after,before", help="which arms run; an unchanged arm's last readings are reused instead (CLAUDE.md, \"Testing tiers\")")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    os.makedirs(a.out, exist_ok=True)
    text = preamble(a.repo)
    warm(a.after, 18520, a.out)
    recs = []
    port = 18531
    for arm, binary in (("after", a.after), ("before", a.before)):
        if arm not in a.arms.split(","):
            continue
        for name in ("0.5b", "1.5b"):
            port += 1
            r = cell(arm, binary, name, text, port, a.out, a.interval)
            recs.append(r)
            print(json.dumps(r), flush=True)
            with open(f"{a.out}/g40-mem.json", "w") as f:
                json.dump(recs, f, indent=1)
    after = [r for r in recs if r["arm"] == "after"]
    v = "PASS" if all(r["verdict"] == "PASS" for r in after) else (
        "VOID" if any(r["verdict"] == "VOID" for r in after) else (
            "PARKED" if all(r["verdict"] in ("PASS", "PARKED") for r in after) else "FAIL"))
    for r in recs:
        if "peak_fp" in r:
            print(f"G40-2 {r['arm']:6} {r['model']}: {r['prompt_tokens']} tokens in {r['wall_s']} s; peak footprint {r['peak_fp']/MiB:.0f} MiB "
                  f"(bound {r['bound_fp']/MiB:.0f}), peak RSS {r['peak_rss']/MiB:.0f} MiB (bound {r['bound_rss']/MiB:.0f}), "
                  f"10 s after {r['after_fp']/MiB:.0f} MiB (bound {r['bound_after']/MiB:.0f}), swap grown {r['swap_grown']/MiB:.0f} MiB -> {r['verdict']}")
        else:
            print(f"G40-2 {r['arm']:6} {r['model']}: {r['verdict']} ({r.get('why', '')})")
    print(f"G40-2 VERDICT {v} (graded on the after arm; the before arm is the comparison)")
    return 0 if v == "PASS" else (2 if v == "VOID" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
