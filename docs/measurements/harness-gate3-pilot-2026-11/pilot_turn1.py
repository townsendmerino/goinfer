#!/usr/bin/env python3
"""Gate 3's by-day pilot (README.md beside this file): post one recorded harness request, turn 1 teacher-forced, the way a
Gate 3 cell would, and turn the times into the night-cost estimates.

  pilot_turn1.py <req-1.json from harness_capture.py> <base url> [--samples 30] [--temp 0.7] [--max-tokens 512]

The request goes 1 + --samples times, greedy first: the first pays the whole prompt's prefill (t_cold), the rest reuse its
prefix (t_warm, their median). Streaming is off and the model name is replaced. Prints the times, the tokens, and the full
and reduced per-cell estimates; writes nothing else.
"""
import argparse, json, statistics, time, urllib.request

TURNS = 10          # a trace's turns, assumed until three traces are recorded
COMBOS = 4          # traces x dialects per cell: three traces, both dialects where the harness speaks both
FR_TURNS = 12       # a free-running session's turns


def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer goinfer",
                                          "anthropic-version": "2023-06-01"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    return time.time() - t0, j


def cell_hours(t_cold, t_warm, seeds, sessions):
    tf = TURNS * (t_cold + seeds * t_warm)       # teacher-forced: each turn's history once cold, then its samples warm
    fr = sessions * FR_TURNS * t_warm            # free-running: each turn a short delta on a reused prefix
    return COMBOS * (tf + fr) / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("req")
    ap.add_argument("base")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    rec = json.load(open(a.req))
    path, body = rec["path"].split("?")[0], dict(rec["body"])
    body["model"], body["stream"] = "pilot", False
    body.pop("stream_options", None)
    key = "max_tokens" if path in ("/v1/messages", "/v1/chat/completions") else "max_output_tokens"
    body[key] = a.max_tokens
    times, outs, ins = [], [], []
    for i in range(1 + a.samples):
        b = dict(body, temperature=0 if i == 0 else a.temp)
        dt, j = post(a.base.rstrip("/"), path, b)
        u = j.get("usage") or {}
        ins.append(u.get("input_tokens") or u.get("prompt_tokens") or 0)
        outs.append(u.get("output_tokens") or u.get("completion_tokens") or 0)
        times.append(dt)
        print(f"request {i}: {dt:.1f} s, {ins[-1]} in, {outs[-1]} out", flush=True)
    t_cold, t_warm = times[0], statistics.median(times[1:])
    print(f"t_cold {t_cold:.1f} s (prompt {ins[0]} tokens), t_warm median {t_warm:.1f} s, "
          f"mean output {statistics.mean(outs):.0f} tokens")
    print(f"per M26-class cell: full (30 seeds, 10 sessions) {cell_hours(t_cold, t_warm, 30, 10):.1f} h, "
          f"reduced (10 seeds, 5 sessions) {cell_hours(t_cold, t_warm, 10, 5):.1f} h")


if __name__ == "__main__":
    main()
