#!/usr/bin/env python3
"""G-SB11b-6 (docs/tasks/task-subsystems-2026-10.md §8, "SB11b"): one goinfer-serve, one prompt, a warm-up, then 8 requests
alternating cost off and on. Each reply's timings.decode_ms over its completion tokens is that request's decode ms per
token; the ratio on/off is taken per consecutive (off, on) pair and reported, not graded.
Usage: cost_on.py <serve binary> <model path> <out.json>. Run it under the exclusive timing lock."""
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

serve, model, out = sys.argv[1:4]
port = 8096
base = f"http://127.0.0.1:{port}"
proc = subprocess.Popen([serve, "-model", model, "-backend", "cuda", "-addr", f"127.0.0.1:{port}"],
                        stdout=open(out + ".serve.log", "w"), stderr=subprocess.STDOUT)


def get(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.load(r)


def chat(name, want_cost):
    body = {"model": name, "temperature": 0, "max_tokens": 128, "seed": 1,
            "messages": [{"role": "user", "content": "Write a short Go function that reverses a string, with a comment."}]}
    if want_cost:
        body["goinfer"] = {"cost": True}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


try:
    deadline = time.time() + 300
    while True:
        try:
            name = get("/v1/models")["data"][0]["id"]
            break
        except Exception:
            if time.time() > deadline or proc.poll() is not None:
                sys.exit("VOID G-SB11b-6: the server did not come up")
            time.sleep(1)
    chat(name, False)  # warm-up, not recorded
    rows = []
    for i in range(8):
        want = i % 2 == 1
        load = os.getloadavg()[0]
        r = chat(name, want)
        t, u = r.get("timings", {}), r["usage"]
        n = u["completion_tokens"]
        rows.append({"cost": want, "load1": load, "completion_tokens": n, "timings": t,
                     "decode_ms_per_token": t["decode_ms"] / n if "decode_ms" in t and n else None,
                     "cost_phases": sorted((r.get("cost") or {}).keys()) if want else None})
    ratios = [rows[i + 1]["decode_ms_per_token"] / rows[i]["decode_ms_per_token"] for i in range(0, 8, 2)
              if rows[i]["decode_ms_per_token"] and rows[i + 1]["decode_ms_per_token"]]
    res = {"model": model, "serve": serve, "rows": rows, "pair_ratios_on_over_off": ratios,
           "median_ratio": statistics.median(ratios) if ratios else None}
    json.dump(res, open(out, "w"), indent=1)
    for row in rows:
        print(f"cost {'on ' if row['cost'] else 'off'} load {row['load1']:.2f}: {row['completion_tokens']} tokens, "
              f"{row['decode_ms_per_token']} ms/token, phases {row['cost_phases']}")
    print(f"G-SB11b-6 (reported, not graded): decode ms/token on/off per pair {['%.4f' % x for x in ratios]}, "
          f"median {res['median_ratio']}")
finally:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
