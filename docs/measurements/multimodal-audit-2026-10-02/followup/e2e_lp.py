import sys, json, time, base64, urllib.request, urllib.error, subprocess, os, signal
S = os.path.dirname(os.path.abspath(__file__)); port = 18141
p = subprocess.Popen([f"{S}/serve-cuda-new", "--addr", f"127.0.0.1:{port}", "--backend", "cuda", "--ctx", "4096", "--model", "/home/francis/models/qwen2.5-vl-3b"], stdout=open(f"{S}/e2e.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
t0 = time.time()
while time.time() - t0 < 120:
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read(); break
    except Exception: time.sleep(1)
png = base64.b64encode(open(f"{S}/redsq.png", "rb").read()).decode()
content = [{"type": "text", "text": "What shapes and colours are in this image? Answer briefly."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}]
def post(extra):
    body = {"model": "qwen2.5-vl-3b", "temperature": 0, "max_tokens": 24, "messages": [{"role": "user", "content": content}], **extra}
    try:
        r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()
try:
    st, b = post({"logprobs": True, "top_logprobs": 2}); r = json.loads(b)
    lp = r["choices"][0]["logprobs"]["content"]
    print("logprobs:", st, "completion_tokens", r["usage"]["completion_tokens"], "entries", len(lp), "tokens", [t["token"] for t in lp][:5], "top2 of first", [round(x["logprob"], 3) for x in lp[0]["top_logprobs"]])
    st, b = post({"logprobs": True, "top_logprobs": 2}); r = json.loads(b); print("repeat (reused %s):" % r["usage"]["prefill_reused_tokens"], "entries", len(r["choices"][0]["logprobs"]["content"]))
    st, b = post({"logprobs": True, "stream": True}); print("stream+logprobs:", st, b.strip()[:160])
    st, b = post({}); print("plain:", st, "logprobs key present:", "logprobs" in json.loads(b)["choices"][0])
finally:
    os.killpg(p.pid, signal.SIGTERM)
