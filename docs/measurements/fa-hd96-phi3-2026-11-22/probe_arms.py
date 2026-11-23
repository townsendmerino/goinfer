#!/usr/bin/env python3
"""Gate 3's untimed arm probe (the hd-96 lane on Phi-3 mini, from OP1-o2's): the depth-2,048 prompt plus an instruction that forces a long reply, greedy, 256 tokens, on the exact lane
(twice, for determinism) and on the hd-96 flash-decode lane (the flag on). Exit 0 when the exact lane repeats itself and the lane's text
DIFFERS from it: the lane's outputs differ from the exact lane's, so identical text over a long reply means the flag did not reach the
lane. No log-probabilities are requested: serve scopes a request that asks for them to exact attention. Usage: probe_arms.py <serve binary> <out dir>."""
import json, os, signal, socket, subprocess, sys, time, urllib.request

BIN, OUT = sys.argv[1], sys.argv[2]
MODEL = os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8126
prompt = json.load(open(os.path.join(HERE, "../../../scripts/prompts.json")))["phi3-mini:2048"]["text"]
os.makedirs(os.path.join(OUT, "serve-logs"), exist_ok=True)


def run(flag):
    log = open(os.path.join(OUT, "serve-logs", f"probe-{flag}.log"), "ab")
    p = subprocess.Popen([BIN, "-model", "bench=" + MODEL, "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", f"-flash-decode-hd96={flag}"],
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
            data=json.dumps({"model": "bench", "messages": [{"role": "user", "content": prompt + "\n\nIgnore everything above. Write a long, detailed short story about a lighthouse keeper, at least 400 words."}], "max_tokens": 256, "temperature": 0}).encode())
        resp = json.load(urllib.request.urlopen(req, timeout=600))
        return resp["choices"][0]["message"]["content"], resp["usage"]["completion_tokens"]
    finally:
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=60)


(exact, n_exact), (lane, n_lane), (exact2, n_exact2) = run("false"), run("true"), run("false")
print(f"probe: exact {n_exact} tokens {exact[-70:]!r}\nprobe: lane  {n_lane} tokens {lane[-70:]!r}")
print(f"probe: exact twice identical: {exact == exact2}; text equal across arms: {exact == lane}")
if exact != exact2:
    print("probe: the exact lane did not repeat itself")
    sys.exit(1)
if n_exact < 200 or n_lane < 200:
    print("probe: a reply under 200 tokens cannot tell the arms apart")
    sys.exit(1)
if exact == lane:
    print("probe: IDENTICAL text on both arms over a long reply")
    sys.exit(1)
print("probe: the arms differ")
