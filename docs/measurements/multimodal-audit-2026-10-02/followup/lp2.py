import sys, json, time, base64, urllib.request, subprocess, os, signal
binp, port, img, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
S = os.path.dirname(os.path.abspath(__file__))
log = open(f"{S}/lp-{tag}.log", "w")
p = subprocess.Popen([binp, "--addr", f"127.0.0.1:{port}"] + sys.argv[5:], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
t0 = time.time()
while time.time() - t0 < 120:
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read(); break
    except Exception: time.sleep(1)
png = base64.b64encode(open(img, "rb").read()).decode()
IMG = [{"type": "text", "text": "What shapes and colours are in this image? Answer briefly."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}]
try:
    for label in ("cold", "reuse"):
        body = {"model": "qwen2.5-vl-3b", "temperature": 0, "max_tokens": 24, "logprobs": True, "top_logprobs": 5, "messages": [{"role": "user", "content": IMG}]}
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120))
        print(label, r["choices"][0]["message"]["content"], r["usage"]["prefill_reused_tokens"], flush=True)
finally:
    time.sleep(1); os.killpg(p.pid, signal.SIGTERM)
