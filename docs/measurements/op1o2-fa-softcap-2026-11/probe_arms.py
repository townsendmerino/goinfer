#!/usr/bin/env python3
"""OP1-o2's untimed arm probe: the same depth-2,048 prompt, greedy, 64 tokens, on the exact lane and on the capped flash-decode lane
(the flag on). Exit 0 when both answer and the texts DIFFER; the lane's outputs differ from the exact lane's, so identical text means the
flag did not reach the lane. Usage: probe_arms.py <serve binary> <out dir>."""
import json, os, signal, socket, subprocess, sys, time, urllib.request

BIN, OUT = sys.argv[1], sys.argv[2]
MODEL = os.path.expanduser("~/models/gemma-2-2b-it-Q8_0.gguf")
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8124
prompt = json.load(open(os.path.join(HERE, "../../../scripts/prompts.json")))["gemma2-2b:2048"]
os.makedirs(os.path.join(OUT, "serve-logs"), exist_ok=True)


def run(flag):
    log = open(os.path.join(OUT, "serve-logs", f"probe-{flag}.log"), "ab")
    p = subprocess.Popen([BIN, "-model", MODEL, "-backend", "cuda", "-quant", "int4", "-addr", f"127.0.0.1:{PORT}", f"-flash-decode-softcap={flag}"],
                         stdout=subprocess.DEVNULL, stderr=log)
    try:
        for _ in range(240):
            try:
                socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            raise SystemExit(f"the server (flag {flag}) never listened")
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"model": "gemma-2-2b-it-Q8_0", "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0}).encode())
        return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
    finally:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=60)


exact, lane = run("false"), run("true")
print(f"probe: exact {exact[:70]!r}\nprobe: lane  {lane[:70]!r}")
if exact == lane:
    print("probe: IDENTICAL text on both arms")
    sys.exit(1)
print("probe: the arms differ")
