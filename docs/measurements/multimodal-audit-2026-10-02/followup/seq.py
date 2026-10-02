import sys, json, time, base64, urllib.request, subprocess, os, signal
binp, port, img, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
args = sys.argv[5:]
S = os.path.dirname(os.path.abspath(__file__))
log = open(f"{S}/seq-{tag}.log", "w")
p = subprocess.Popen([binp, "--addr", f"127.0.0.1:{port}"] + args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
t0 = time.time(); up = False
while time.time() - t0 < 120:
    if p.poll() is not None: print("exited", p.returncode); break
    try: urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read(); up = True; break
    except Exception: time.sleep(1)
print(tag, "up", up, f"{time.time()-t0:.1f}s", flush=True)
png = base64.b64encode(open(img, "rb").read()).decode()
def ask(content, mt=24):
    body = {"model": "qwen2.5-vl-3b", "temperature": 0, "max_tokens": mt, "messages": [{"role": "user", "content": content}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120))
    return r["choices"][0]["message"]["content"], r["usage"].get("prefill_reused_tokens"), r["choices"][0]["finish_reason"]
IMG = [{"type": "text", "text": "What shapes and colours are in this image? Answer briefly."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}]
try:
    for label, c in [("A cold", IMG), ("A rep1", IMG), ("A rep2", IMG), ("T evict", "Say hi."), ("A cold2", IMG), ("A rep3", IMG), ("T evict", "Say hi."), ("A cold3", IMG), ("A rep4", IMG)]:
        print(f"{tag} {label:8s}", ask(c), flush=True)
finally:
    os.killpg(p.pid, signal.SIGTERM)
