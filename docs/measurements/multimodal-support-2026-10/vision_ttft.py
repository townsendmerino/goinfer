#!/usr/bin/env python3
"""New-image (or new-clip) time to first token, served, for S7 and S13-lite (docs/tasks/task-multimodal-support-2026-10.md).

Every request, warm-up included, carries media no server has seen: the base image (or WAV) with that request's own
pseudorandom least-significant-bit pattern (fresh_png / fresh_wav, the method of scripts/bench_peer.py's fresh_image), so
the size and content are the base's and every byte-hash cache misses. TTFT is the wall time from sending a streaming
request to its first non-empty content delta.

A plan is a JSON file: {"rounds": R, "per_round": K, "rotate": bool, "cells": [cell, ...]}, a cell being
  {"cell": label, "engine": label, "api": "openai"|"ollama", "cmd": [argv], "port": N, "model": served name or "" (ask
   /v1/models), "media": "image"|"audio", "base": path, "prompt": text, "env": {extra env}}.
Each round runs every cell once (rotated by the round number when "rotate", for same-session interleaving): start the
server, wait for it, one warm-up request, K timed requests, stop it. Writes <out>/requests.jsonl, <out>/<cell>-<engine>-r<n>.log
(the server's output) and prints a per-cell summary with the median TTFT and the 5 s bar.

Run: python3 vision_ttft.py <plan.json> <out dir>
"""
import base64
import io
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import wave

BAR_S = 5.0


def fresh_png(path, i):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    r, g, b = im.split()
    rng = random.Random(1_000_003 * (i + 1))
    bits = bytes(rng.getrandbits(1) for _ in range(im.width * im.height))
    r = Image.frombytes("L", im.size, bytes(p ^ q for p, q in zip(r.tobytes(), bits)))
    out = io.BytesIO()
    Image.merge("RGB", (r, g, b)).save(out, format="PNG")
    return out.getvalue()


def fresh_wav(path, i):
    with wave.open(path) as w:
        params, frames = w.getparams(), bytearray(w.readframes(w.getnframes()))
    rng = random.Random(2_000_003 * (i + 1))
    for k in range(0, len(frames), 2):  # 16-bit LE: flip the low bit of the low byte
        frames[k] ^= rng.getrandbits(1)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(bytes(frames))
    return out.getvalue()


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def wait_up(c, proc, deadline_s=900):
    url = f"http://127.0.0.1:{c['port']}" + ("/api/tags" if c["api"] == "ollama" else "/v1/models")
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if proc.poll() is not None:
            return False
        try:
            st, _ = get(url)
            if st == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def served_model(c):
    if c.get("model"):
        return c["model"]
    _, body = get(f"http://127.0.0.1:{c['port']}/v1/models")
    return json.loads(body)["data"][0]["id"]


def request(c, model, media):
    b64 = base64.b64encode(media).decode()
    if c["api"] == "ollama":
        url = f"http://127.0.0.1:{c['port']}/api/chat"
        body = {"model": model, "stream": True, "options": {"temperature": 0, "num_predict": 8},
                "messages": [{"role": "user", "content": c["prompt"], "images": [b64]}]}
    else:
        url = f"http://127.0.0.1:{c['port']}/v1/chat/completions"
        part = ({"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}} if c["media"] == "audio"
                else {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}})
        body = {"model": model, "stream": True, "max_tokens": 8, "temperature": 0,
                "messages": [{"role": "user", "content": [part, {"type": "text", "text": c["prompt"]}]}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text = None, ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
                if line == "[DONE]":
                    break
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if c["api"] == "ollama":
                piece = (ev.get("message") or {}).get("content") or ""
            else:
                ch = (ev.get("choices") or [{}])[0]
                piece = (ch.get("delta") or {}).get("content") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            text += piece
    return ttft, text


def main():
    plan, out = json.load(open(sys.argv[1])), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    jl = open(os.path.join(out, "requests.jsonl"), "a")
    cells, nreq = plan["cells"], 0
    results = {}
    for rnd in range(plan["rounds"]):
        order = cells[rnd % len(cells):] + cells[:rnd % len(cells)] if plan.get("rotate") else cells
        for c in order:
            key = f"{c['cell']} | {c['engine']}"
            logp = os.path.join(out, f"{c['cell']}-{c['engine']}-r{rnd + 1}.log".replace(" ", "_").replace("/", "_"))
            env = dict(os.environ, **c.get("env", {}))
            print(f"[{time.strftime('%H:%M:%S')}] round {rnd + 1}: {key}: {' '.join(c['cmd'])}", flush=True)
            with open(logp, "w") as lf:
                proc = subprocess.Popen(c["cmd"], stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                        start_new_session=True)
            try:
                t_load = time.time()
                if not wait_up(c, proc):
                    print(f"  !! server did not come up; see {logp}", flush=True)
                    jl.write(json.dumps({"cell": c["cell"], "engine": c["engine"], "round": rnd + 1, "error": "no server"}) + "\n")
                    continue
                model = served_model(c)
                print(f"  up in {time.time() - t_load:.0f} s, model {model!r}; load {os.getloadavg()[0]:.2f}", flush=True)
                for k in range(plan["per_round"] + 1):
                    media = fresh_wav(c["base"], nreq) if c["media"] == "audio" else fresh_png(c["base"], nreq)
                    nreq += 1
                    rec = {"cell": c["cell"], "engine": c["engine"], "round": rnd + 1, "req": k, "warmup": k == 0,
                           "media_index": nreq - 1, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    try:
                        ttft, text = request(c, model, media)
                        rec.update(ttft_s=ttft, text=text)
                        if k > 0 and ttft is not None:
                            results.setdefault(key, []).append(ttft)
                        print(f"  {'warm-up' if k == 0 else 'timed  '} {k}: TTFT {ttft if ttft is None else round(ttft, 3)} s {text[:60]!r}", flush=True)
                    except Exception as e:
                        rec["error"] = str(e)
                        print(f"  !! request {k}: {e}", flush=True)
                    jl.write(json.dumps(rec) + "\n")
                    jl.flush()
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                time.sleep(3)
    print("\n| cell | engine | timed requests | median TTFT (s) | under the 5 s bar |\n|---|---|---|---|---|")
    for c in cells:
        key = f"{c['cell']} | {c['engine']}"
        xs = results.get(key, [])
        med = statistics.median(xs) if xs else None
        print(f"| {c['cell']} | {c['engine']} | {len(xs)} | {'-' if med is None else f'{med:.2f}'} | "
              f"{'-' if med is None else ('yes' if med < BAR_S else 'no')} |")


if __name__ == "__main__":
    main()
