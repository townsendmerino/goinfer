import sys, json, time, base64, urllib.request, subprocess, os, signal
name, port, img = sys.argv[1], int(sys.argv[2]), sys.argv[3]
args = sys.argv[4:]
S = os.path.dirname(os.path.abspath(__file__))
log = open(f"{S}/{name}.log", "w")
p = subprocess.Popen([os.environ.get("PROBE_BIN", f"{S}/serve-cuda"), "--addr", f"127.0.0.1:{port}"] + args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open(f"{S}/{name}.pid", "w").write(str(p.pid))
t0 = time.time()
up = False
while time.time() - t0 < 560:
    if p.poll() is not None: print("server exited", p.returncode); break
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read(); up = True; break
    except Exception: time.sleep(1)
print(f"[{name}] up={up} after {time.time()-t0:.1f}s", flush=True)
if up:
    models = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))
    print("models:", json.dumps(models)[:300])
    png = base64.b64encode(open(img, "rb").read()).decode()
    body = {"model": models["data"][0]["id"], "temperature": 0, "max_tokens": 24,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What shapes and colours are in this image? Answer briefly."},
                          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}]}]}
    for i in range(2):
        t1 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
            print('RAW', json.dumps(r)[:900])
            print(f"req{i}: {time.time()-t1:.1f}s finish={r['choices'][0].get('finish_reason')} usage={r.get('usage')} reply={r['choices'][0]['message']['content']!r}", flush=True)
        except Exception as e:
            print(f"req{i}: ERROR {e}", flush=True); 
            try: print(e.read()[:300])
            except Exception: pass
            break
os.killpg(p.pid, signal.SIGTERM)
time.sleep(2)
print("--- key log lines")
for l in open(f"{S}/{name}.log"):
    if any(k in l.lower() for k in ("vision", "decode path", "backend", "loaded", "resident", "image", "warn", "error", "context")): print("  ", l.rstrip()[:260])
