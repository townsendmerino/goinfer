import json, os, signal, socket, subprocess, sys, time, urllib.request
BIN, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts"))
prompts = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts/prompts.json")))
key = next(k for k in prompts if k.startswith("G20:") and k.endswith(":128")) if any(k.startswith("G20:") for k in prompts) else None
text = prompts[key]["text"] if key else "Write a short essay about lighthouses."
log = open(os.path.join(OUT, "serve.log"), "wb")
p = subprocess.Popen([BIN, "-model", "bench=" + MODEL, "-backend", "cpu", "-ctx", "4096", "-addr", "127.0.0.1:18211"], stdout=subprocess.DEVNULL, stderr=log)
try:
    for _ in range(300):
        if p.poll() is not None: sys.exit("server exited")
        try: socket.create_connection(("127.0.0.1", 18211), timeout=1).close(); break
        except OSError: time.sleep(1)
    body = {"model": "bench", "messages": [{"role": "user", "content": text}], "max_tokens": 64, "temperature": 0, "seed": 1,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request("http://127.0.0.1:18211/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                events.append((time.perf_counter(), json.loads(line[5:])))
    usage = next((e["usage"] for _, e in events if e.get("usage")), None)
    tok_ev = []
    for t, e in events:
        ch = e.get("choices") or []
        if not ch or ch[0].get("finish_reason") is not None: continue
        d = ch[0].get("delta", {})
        if d.get("role") is not None and not d.get("content"): continue
        tok_ev.append((t, d))
    keys = {}
    for _, d in tok_ev:
        for k in d: keys[k] = keys.get(k, 0) + 1
    print(json.dumps({"prompt_key": key, "usage": usage, "events_total": len(events), "token_events": len(tok_ev),
                      "with_content": sum(1 for _, d in tok_ev if d.get("content")), "delta_keys": keys,
                      "first_deltas": [d for _, d in tok_ev[:3]],
                      "gaps_ms": [round((tok_ev[i][0]-tok_ev[i-1][0])*1000, 1) for i in range(1, min(len(tok_ev), 12))]}, indent=1))
finally:
    p.send_signal(signal.SIGTERM); p.wait(timeout=60); log.close()
