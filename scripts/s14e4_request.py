#!/usr/bin/env python3
"""G-S14e4's request: one WAV as an input_audio part of one chat message, greedy, to a served Voxtral (or any audio model). Usage: s14e4_request.py <wav> <out.json> <port>.
Writes {reply, finish, usage, seconds}."""
import base64, json, sys, time, urllib.request
wav, out, port = sys.argv[1], sys.argv[2], sys.argv[3]
b64 = base64.b64encode(open(wav, "rb").read()).decode()
body = {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}], "max_tokens": 400, "temperature": 0}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600))
c = r["choices"][0]
json.dump({"reply": c["message"]["content"], "finish": c["finish_reason"], "usage": r["usage"], "seconds": time.time() - t}, open(out, "w"))
print(f"{time.time()-t:.1f}s", r["usage"], c["finish_reason"], repr(c["message"]["content"][:300]))
