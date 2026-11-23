#!/usr/bin/env python3
"""Gate 3's served precondition by launch count (README, gate-3 probe addendum): serve Phi-3 mini twice, -flash-decode-hd96 and =false,
send each the depth-2,048 prompt for 64 greedy tokens with goinfer's cost report, and read the decode phase's kernel launches by
operation. Exit 0 when the lane arm's decode launches a flash-decode kernel and the exact arm's does not.
Usage: probe_lane_cost.py <serve binary> <out dir>."""
import json, os, signal, socket, subprocess, sys, time, urllib.request
BIN, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
PORT = 8127
prompt = json.load(open(os.path.join(HERE, "../../../scripts/prompts.json")))["phi3-mini:2048"]["text"]
os.makedirs(os.path.join(OUT, "serve-logs"), exist_ok=True)

def run(flag):
    path = os.path.join(OUT, "serve-logs", f"probe-cost-{flag}.log")
    log = open(path, "wb")
    p = subprocess.Popen([BIN, "-model", "bench=" + MODEL, "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", f"-flash-decode-hd96={flag}"],
                         stdout=subprocess.DEVNULL, stderr=log)
    try:
        for _ in range(240):
            if p.poll() is not None:
                raise SystemExit(f"the server ({flag}) exited early; see {path}")
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", method="POST", headers={"Content-Type": "application/json"},
            data=json.dumps({"model": "bench", "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0,
                             "goinfer": {"cost": True}}).encode())
        r = json.load(urllib.request.urlopen(req, timeout=600))
    finally:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=60)
        log.close()
    dec = (r.get("cost") or {}).get("decode") or {}
    return {op: c.get("launches", 0) for op, c in (dec.get("ops") or {}).items()}, dec.get("tokens"), r["usage"]["prompt_tokens"]

lane, nl, pl = run("true")
exact, ne, pe = run("false")
# The lane's combine is a named pipeline (faCombine); its partial kernels sit in a table pipeName does not name ("?"). The exact path's
# decode attention is bAttn. Each arm must launch its own kernel on every layer's decode token and never the other's.
print(f"probe-cost: lane  prompt {pl} decode tokens {nl} ops {sorted(lane.items())}")
print(f"probe-cost: exact prompt {pe} decode tokens {ne} ops {sorted(exact.items())}")
print(f"probe-cost: faCombine lane {lane.get('faCombine', 0)} exact {exact.get('faCombine', 0)}; bAttn lane {lane.get('bAttn', 0)} exact {exact.get('bAttn', 0)}")
if not lane.get("faCombine") or lane.get("bAttn") or not exact.get("bAttn") or exact.get("faCombine"):
    print("probe-cost: the lane arm did not launch the lane, or the exact arm did")
    sys.exit(1)
print("probe-cost: the lane arm launches the lane, the exact arm does not")
