#!/usr/bin/env python3
"""Gate 3's served precondition (README beside this file): serve Phi-3 mini twice, -flash-decode-hd96 and =false, with
GOINFER_ATTNFA_DEBUG=1, send each the depth-2,048 prompt for 16 greedy tokens, and count the [ATTNFA] lines the encoder logs each time
it dispatches attention_fa on a layer. Exit 0 when the lane arm logs hd=96 dispatches and the exact arm logs none.
Usage: probe_lane_metal.py <serve binary> <out dir>."""
import json, os, signal, socket, subprocess, sys, time, urllib.request
BIN, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
PORT = 8127
prompt = json.load(open(os.path.join(HERE, "../../../scripts/prompts.json")))["phi3-mini:2048"]["text"]
os.makedirs(os.path.join(OUT, "serve-logs"), exist_ok=True)


def run(flag):
    path = os.path.join(OUT, "serve-logs", f"probe-{flag}.log")
    with open(path, "wb") as log:
        p = subprocess.Popen([BIN, "-model", "bench=" + MODEL, "-backend", "metal", "-addr", f"127.0.0.1:{PORT}", f"-flash-decode-hd96={flag}"],
                             stdout=subprocess.DEVNULL, stderr=log, env=dict(os.environ, GOINFER_ATTNFA_DEBUG="1"))
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
                data=json.dumps({"model": "bench", "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "temperature": 0}).encode())
            r = json.load(urllib.request.urlopen(req, timeout=900))
        finally:
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=60)
    lines = [ln for ln in open(path, errors="replace") if "[ATTNFA]" in ln]
    return len(lines), sum("hd=96" in ln for ln in lines), r["usage"]["prompt_tokens"]


lane, lane96, pl = run("true")
exact, exact96, pe = run("false")
print(f"probe: lane  prompt {pl} tokens, [ATTNFA] lines {lane} (hd=96: {lane96})")
print(f"probe: exact prompt {pe} tokens, [ATTNFA] lines {exact} (hd=96: {exact96})")
if not lane96 or exact:
    print("probe: the lane arm did not dispatch attention_fa at hd 96, or the exact arm dispatched it")
    sys.exit(1)
print("probe: the lane arm dispatches attention_fa at hd 96, the exact arm does not")
