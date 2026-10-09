#!/usr/bin/env python3
"""G-S14c4's record-only arm D (docs/tasks/task-multimodal-support-2026-10.md): Gemma 4 E4B asked to transcribe the speech and reply with only the transcription on each clip through a running serve (input_audio), greedy, 160 tokens at most.
Writes {id: reply} to <out>. Usage: s14c4_gemma.py <port> <wav dir> <out.json> [max clips]"""
import base64, json, sys, time, urllib.request
port, data, out = sys.argv[1:4]; lim = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ids = sorted(json.load(open(f"{data}/refs.json")))[: lim or None]
res, t0 = {}, time.time()
for k, cid in enumerate(ids):
    b64 = base64.b64encode(open(f"{data}/{cid}.wav", "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}, {"type": "text", "text": "Transcribe the speech in this audio exactly. Reply with only the transcription, nothing else."}]}], "max_tokens": 160, "temperature": 0}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
        res[cid] = r["choices"][0]["message"]["content"] or ""
    except Exception as e:
        res[cid] = ""; print(f"  {cid}: request failed: {e}", flush=True)
    print(f"[s14c4 gemma {time.time() - t0:5.0f}s] {k + 1}/{len(ids)} {cid}: {res[cid][:90]!r}", flush=True)
json.dump(res, open(out, "w"), indent=0)
