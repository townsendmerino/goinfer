#!/usr/bin/env python3
"""replay.py OUTJSON BODY[:override] ... — POST captured opencode bodies to the server on 43299 in order, greedy, non-streamed, with top-20 logprobs.
A body path ends in ':gen' when its reply is to be recorded (max_tokens 64); other bodies run with max_tokens 1."""
import json, sys, urllib.request, time
out = sys.argv[1]; res = []
for spec in sys.argv[2:]:
    path, _, mode = spec.partition(":")
    b = json.load(open(path))
    b.pop("stream_options", None); b["stream"] = False
    b["temperature"] = 0; b["max_tokens"] = 256 if mode == "gen256" else 64 if mode == "gen" else 1
    if mode in ("gen", "lp"): b["logprobs"] = True; b["top_logprobs"] = 20
    r = urllib.request.Request("http://127.0.0.1:43299/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(r, timeout=600)); dt = time.time() - t
    ch = d["choices"][0]
    print(f"{path} mode={mode or 'prime'} {dt:.1f}s prompt_tokens={d['usage']['prompt_tokens']} finish={ch['finish_reason']}", flush=True)
    res.append({"path": path, "mode": mode, "secs": dt, "usage": d["usage"], "message": ch["message"], "logprobs": ch.get("logprobs"), "finish": ch["finish_reason"]})
json.dump(res, open(out, "w"))
