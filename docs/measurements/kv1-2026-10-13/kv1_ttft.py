#!/usr/bin/env python3
"""KV1's G5b, time to first token, new against main (docs/measurements/kv1-2026-10-13.md).

One request at a time. A block is one freshly started server per arm, in alternating order (new old, old new, ...): a
warm request, then three long prompts that differ near their start, each timed by the client from just before the
request is written to the first streamed content event. The method is scripts/bench_prefill_stall.py's solo arm, which
is Metal-only; this runs either backend with two binaries.

  kv1_ttft.py OUT.json --backend cuda --new BIN --old BIN --model PATH --name 1.5B [--blocks 12] [--prompt-words 2900]
"""
import argparse, json, os, signal, socket, subprocess, sys, time, urllib.request

PORT = 8097
PARA = ("The quick survey of rate limiters begins with the token bucket, which refills at a steady rate and lets a "
        "burst spend what has accumulated; the sliding window log keeps every timestamp and counts those inside the "
        "window; the fixed window counter is cheaper but double-admits at the boundary; and the leaky bucket smooths "
        "output to a constant drain. Section {i} weighs these against a distributed deployment where clocks drift. ")


def long_prompt(words, tag):
    out, i = [], 0
    while len(" ".join(out).split()) < words:
        out.append(PARA.format(i=i))
        i += 1
    return f"[{tag}] Read the following notes, then answer in one sentence: which approach do they favour?\n\n" + "".join(out)


def wait_port(proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.25)
    return False


def request(content, max_tokens):
    """Returns (seconds to the first content event, prompt_tokens the server reported)."""
    body = {"model": "bench", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first, prompt_tokens = None, None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            if ev.get("usage"):
                prompt_tokens = ev["usage"].get("prompt_tokens")
            ch = (ev.get("choices") or [{}])[0]
            if first is None and (ch.get("delta") or {}).get("content"):
                first = time.perf_counter() - t0
    return first, prompt_tokens


def cell(binary, a, block, log):
    argv = [binary, "-model", f"bench={a.model}", "-backend", a.backend, "-require-backend", "-addr", f"127.0.0.1:{PORT}",
            "-quant", "int4"] + a.serve_args.split()
    with open(log, "a") as sink:
        sink.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
        sink.flush()
        proc = subprocess.Popen(argv, stdout=sink, stderr=sink, preexec_fn=os.setsid)
        try:
            if not wait_port(proc, a.load_timeout):
                return {"error": "the server did not come up"}
            request("warm", 4)
            ttft, toks = [], []
            for j in range(3):
                t, n = request(long_prompt(a.prompt_words, f"kv1-b{block}-n{j}"), 8)
                if t is None:
                    return {"error": f"prompt {j} streamed no content"}
                ttft.append(round(t, 4))
                toks.append(n)
            return {"ttft_s": ttft, "prompt_tokens": toks}
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--prompt-words", type=int, default=2500)
    ap.add_argument("--serve-args", default="")
    ap.add_argument("--load-timeout", type=float, default=420)
    ap.add_argument("--log-dir", default=".")
    a = ap.parse_args()
    cells, t_start = [], time.time()
    for b in range(a.blocks):
        order = ["new", "old"] if b % 2 == 0 else ["old", "new"]
        for arm in order:
            r = cell(a.new if arm == "new" else a.old, a, b, os.path.join(a.log_dir, f"ttft-{a.backend}-{a.name}-{arm}.log"))
            r.update(arm=arm, block=b, model=a.name, backend=a.backend, at=time.strftime("%H:%M:%S"))
            cells.append(r)
            json.dump({"args": vars(a), "cells": cells}, open(a.out, "w"), indent=1)
            # the heartbeat: one line a cell, with the count and the elapsed time (cells differ little in cost)
            done, total = len(cells), a.blocks * 2
            el = time.time() - t_start
            print(f"[kv1-ttft {a.backend} {a.name}] cell {done}/{total} block {b} {arm}: {r.get('ttft_s', r.get('error'))} "
                  f"({r.get('prompt_tokens')}) elapsed {el/60:.1f} min, about {el/done*(total-done)/60:.1f} min left", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
