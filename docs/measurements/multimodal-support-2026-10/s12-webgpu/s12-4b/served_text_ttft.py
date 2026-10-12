#!/usr/bin/env python3
"""G-S12.4b-2d: served time to first token of a long text prompt under --backend webgpu, two serve
binaries interleaved (docs/tasks/task-multimodal-support-2026-10.md, "S12.4b").

    served_text_ttft.py OUT.json --new BIN --old BIN --model PATH --prompts scripts/prompts.json \
        --key 1.5B --depths 512,2048 --rounds 5 --log-dir DIR

Each round starts each binary once (order alternating by round), sends a discarded warm-up, then one
streamed chat request per depth and times the first content event from just before the request is
written. A request's prompt is the token-calibrated text for its depth behind a prefix that is unique
to the (round, depth) and the same for both arms, so no request reuses another's prefix and the two
arms' replies can be compared: they must be equal, since the two prefills give the same logits.

It grades nothing by exit code: 0 means every cell ran. The verdict it prints is the registered rule
read on the medians (ship at <= 0.98, park to 1.00, off above).
"""
import argparse, json, os, platform, signal, statistics, subprocess, sys, time, urllib.request

PORT = 18652
T0 = time.time()


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:5.0f}s] {msg}", flush=True)


def start(binary, model, log_path):
    argv = [binary, "-model", f"bench={model}", "-backend", "webgpu", "-addr", f"127.0.0.1:{PORT}", "-require-backend"]
    log = open(log_path, "ab")
    log.write(f"\n=== {time.strftime('%F %T')} {' '.join(argv)}\n".encode())
    log.flush()
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{binary} exited with {proc.returncode} before it was ready; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    stop(proc)
    raise RuntimeError(f"{binary} was not ready after 900 s; see {log_path}")


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def request(prompt, ngen):
    """-> (seconds to the first content event, the reply's text)."""
    body = json.dumps({"model": "bench", "stream": True, "max_tokens": ngen, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, text = None, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = d.get("choices") or []
            if not ch:
                continue
            piece = (ch[0].get("delta") or {}).get("content")
            if piece:
                if first is None:
                    first = time.time() - t0
                text.append(piece)
    if first is None:
        raise RuntimeError("the reply had no content event")
    return first, "".join(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--key", required=True, help="the model's key in the prompts file, e.g. 1.5B")
    ap.add_argument("--depths", default="512,2048")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ngen", type=int, default=16)
    ap.add_argument("--log-dir", required=True)
    a = ap.parse_args()
    for root in ("/Volumes/", "/srv/models"):
        if os.path.realpath(a.model).startswith(root):
            sys.exit(f"refusing to measure from {root}: copy the model to ~/models first")
    depths = [int(d) for d in a.depths.split(",")]
    prompts = json.load(open(a.prompts))
    texts = {d: prompts[f"{a.key}:{d}"] for d in depths}
    warm = prompts[f"{a.key}:128"]["text"]
    os.makedirs(a.log_dir, exist_ok=True)
    bins = {"new": a.new, "old": a.old}
    rec = {"schema": 1, "gate": "G-S12.4b-2d", "host": platform.node(), "uname": " ".join(platform.uname()[:3]),
           "started": time.strftime("%F %T %Z"), "model": a.model, "bins": bins,
           "revs": {k: (open(v + ".rev").read().strip() if os.path.exists(v + ".rev") else "") for k, v in bins.items()},
           "depths": {str(d): texts[d]["tokens"] for d in depths}, "rounds": a.rounds, "cells": []}
    say(f"{a.rounds} rounds x 2 arms x {len(depths)} depths ({a.depths} tokens); model {a.model}")
    for r in range(a.rounds):
        for arm in (["new", "old"] if r % 2 == 0 else ["old", "new"]):
            say(f"round {r + 1} of {a.rounds}, {arm}: starting {os.path.basename(bins[arm])} (load {os.getloadavg()[0]:.2f})")
            proc = start(bins[arm], a.model, os.path.join(a.log_dir, f"serve-{arm}.log"))
            try:
                request(f"Warm-up {r:02d}. " + warm, 4)
                for di, d in enumerate(depths):
                    load = os.getloadavg()[0]
                    t, text = request(f"Session {r * 10 + di:04d}. " + texts[d]["text"], a.ngen)
                    rec["cells"].append({"round": r, "arm": arm, "depth": d, "ttft_s": t, "text": text, "load1": load})
                    say(f"round {r + 1} of {a.rounds}, {arm}, {d} tokens: TTFT {t:.2f} s")
            finally:
                stop(proc)
            json.dump(rec, open(a.out, "w"), indent=1)
    print()
    for d in depths:
        new = {c["round"]: c for c in rec["cells"] if c["arm"] == "new" and c["depth"] == d}
        old = {c["round"]: c for c in rec["cells"] if c["arm"] == "old" and c["depth"] == d}
        ratios = [new[r]["ttft_s"] / old[r]["ttft_s"] for r in range(a.rounds)]
        same = sum(new[r]["text"] == old[r]["text"] for r in range(a.rounds))
        med = statistics.median(ratios)
        verdict = "SHIP" if med <= 0.98 else ("PARK" if med <= 1.00 else "OFF")
        ns = " ".join("%.2f" % new[r]["ttft_s"] for r in range(a.rounds))
        os_ = " ".join("%.2f" % old[r]["ttft_s"] for r in range(a.rounds))
        rs = " ".join("%.3f" % x for x in ratios)
        print(f"RESULT {d} tokens: new {ns} s; old {os_} s; ratios {rs}; median {med:.3f} -> {verdict}; "
              f"replies equal in {same} of {a.rounds} rounds")
    rec["finished"] = time.strftime("%F %T %Z")
    json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
