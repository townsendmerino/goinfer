import json, os, signal, socket, subprocess, sys, time, urllib.request
BIN, OUT = sys.argv[1], sys.argv[2]
REPO = os.path.expanduser("~/goinfer-bench/phi3-gap-wt")
text = json.load(open(os.path.join(REPO, "scripts/prompts.json")))["phi3-mini:128"]["text"]
MODEL = os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
log = open(os.path.join(OUT, "serve-cost.log"), "wb")
p = subprocess.Popen([BIN, "-model", "bench=" + MODEL, "-backend", "cuda", "-ctx", "4096", "-addr", "127.0.0.1:18234"], stdout=subprocess.DEVNULL, stderr=log)
try:
    for _ in range(300):
        if p.poll() is not None: sys.exit("server exited")
        try: socket.create_connection(("127.0.0.1", 18234), timeout=1).close(); break
        except OSError: time.sleep(1)
    for i in range(2):
        body = {"model": "bench", "stream": False, "max_tokens": 64, "temperature": 0, "seed": 1, "goinfer": {"cost": True},
                "messages": [{"role": "user", "content": text}]}
        req = urllib.request.Request("http://127.0.0.1:18234/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        print(json.dumps({k: r.get(k) for k in ("usage", "timings", "cost")}, indent=1)[:6000])
finally:
    p.send_signal(signal.SIGTERM); p.wait(timeout=60); log.close()
