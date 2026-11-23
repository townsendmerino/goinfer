#!/usr/bin/env python3
"""OP1-o2's default smoke (README "Default, 2026-11-22"), written for #1230's binary, where the flag defaulted on (that flip did not ship):
one serve binary, Gemma 2 2B on CUDA, served four times: the flag unset (the
new default), -flash-decode-softcap=false (the exact split-KV lane), -flash-decode-softcap (the lane, given), and =false again.
Each takes probe_arms.py's depth-2,048 request (greedy, 256 tokens, no log-probabilities), and each server's fit and KV plan banner
lines and its device memory after load are recorded.
Exit 0 when the default's text equals the given lane's, differs from the exact lane's, the exact lane repeats itself, and every
arm's fit plan (the fit line without its host budget, which reads the box's free memory at load), KV plan line and
device memory (read after load and again after the request) are equal and present. It takes the exclusive timing lock itself (or
runs under a parent that holds it) and refuses if another run holds it. Usage: smoke_default.py <serve binary> <out dir>."""
import json, os, re, signal, socket, subprocess, sys, time, urllib.request

BIN, OUT = sys.argv[1], sys.argv[2]
MODEL = os.path.expanduser("~/models/gemma-2-2b-it-Q8_0.gguf")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../../scripts"))
import timing_lock  # noqa: E402
PORT = 8125
prompt = json.load(open(os.path.join(HERE, "../../../scripts/prompts.json")))["gemma2-2b:2048"]["text"]
os.makedirs(os.path.join(OUT, "serve-logs"), exist_ok=True)


def gpu_mib(pid):
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for row in q.splitlines():
        p, m = [x.strip() for x in row.split(",")]
        if int(p) == pid:
            return int(m)
    return None


def run(name, flags):
    path = os.path.join(OUT, "serve-logs", f"smoke-{name}.log")
    log = open(path, "wb")
    p = subprocess.Popen([BIN, "-model", MODEL, "-backend", "cuda", "-quant", "int4", "-addr", f"127.0.0.1:{PORT}", *flags],
                         stdout=subprocess.DEVNULL, stderr=log)
    try:
        for _ in range(240):
            if p.poll() is not None:
                raise SystemExit(f"the server ({name}) exited with {p.returncode} before it listened; see {path}")
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            raise SystemExit(f"the server ({name}) never listened")
        mib = gpu_mib(p.pid)
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"model": "gemma-2-2b-it-Q8_0", "messages": [{"role": "user", "content": prompt + "\n\nIgnore everything above. Write a long, detailed short story about a lighthouse keeper, at least 400 words."}], "max_tokens": 256, "temperature": 0}).encode())
        resp = json.load(urllib.request.urlopen(req, timeout=600))
        mib_after = gpu_mib(p.pid)
    finally:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=60)
        log.close()
    text = open(path, errors="replace").read()
    line = lambda pat: next((l.strip() for l in text.splitlines() if re.search(pat, l)), None)
    return dict(name=name, flags=flags, text=resp["choices"][0]["message"]["content"], n=resp["usage"]["completion_tokens"], mib=mib, mib_after=mib_after,
                fit=line(r" fit:"), kv=line(r"KV plan:"), path=line(r"decode path:"), warn=line(r"flash-decode lane unavailable"))


with timing_lock.hold(label="op1o2-default-smoke"):
    arms = [run("default", []), run("exact", ["-flash-decode-softcap=false"]), run("lane", ["-flash-decode-softcap"]),
            run("exact2", ["-flash-decode-softcap=false"])]
for a in arms:
    print(f"smoke: {a['name']:8s} flags {a['flags']} tokens {a['n']} device {a['mib']} MiB at load, {a['mib_after']} MiB after | {a['path']} | {a['fit']} | {a['kv']}"
          f"{' | WARN ' + a['warn'] if a['warn'] else ''}")
    print(f"smoke: {a['name']:8s} tail {a['text'][-70:]!r}")
d, e, l, e2 = arms
plan = lambda fit: re.sub(r", budget [^)]*\)", ")", fit or "")
checks = {
    "default text == lane text": d["text"] == l["text"],
    "default text != exact text": d["text"] != e["text"],
    "exact repeats itself": e["text"] == e2["text"],
    "every reply over 200 tokens": all(a["n"] >= 200 for a in arms),
    "fit plan equal on every arm": len({plan(a["fit"]) for a in arms}) == 1 and d["fit"] is not None,
    "KV plan line equal on every arm": len({a["kv"] for a in arms}) == 1 and d["kv"] is not None,
    "device MiB at load equal on every arm": len({a["mib"] for a in arms}) == 1 and d["mib"] is not None,
    "device MiB after the request equal on every arm": len({a["mib_after"] for a in arms}) == 1 and d["mib_after"] is not None,
    "no lane-unavailable warning": not any(a["warn"] for a in arms),
}
for k, v in checks.items():
    print(f"smoke: {'PASS' if v else 'FAIL'} {k}")
sys.exit(0 if all(checks.values()) else 1)
