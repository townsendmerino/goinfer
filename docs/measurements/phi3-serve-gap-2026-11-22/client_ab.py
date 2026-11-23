import json, os, signal, socket, subprocess, sys, time
REPO = os.path.expanduser("~/goinfer-bench/phi3-gap-wt")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import bench_peer as bp
BIN, OUT = sys.argv[1], sys.argv[2]
GO = os.path.expanduser("~/goinfer-bench/tmp/nobara-perf-phi3/sseclient/sseclient")
MODEL = os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
PF = os.path.join(REPO, "scripts/prompts.json")
key = "phi3-mini:128"
text = json.load(open(PF))[key]["text"]
log = open(os.path.join(OUT, "serve.log"), "wb")
p = subprocess.Popen([BIN, "-model", "bench=" + MODEL, "-backend", "cuda", "-ctx", "4096", "-addr", "127.0.0.1:18233"], stdout=subprocess.DEVNULL, stderr=log)
url = "http://127.0.0.1:18233/v1/chat/completions"
def py():
    payload = {"model": "bench", "stream": True, "max_tokens": 64, "stream_options": {"include_usage": True},
               "messages": [{"role": "user", "content": text}], "temperature": 0, "seed": 1}
    iv, tf, tl, ch, rep = bp.post_stream(url, payload, bp.parse_openai)
    return (rep - 1) / (tl - tf), iv + 1, rep
def go():
    out = subprocess.run([GO, url, PF, key], capture_output=True, text=True).stdout.split()
    return float(out[out.index("rate") + 1]), int(out[2]), int(out[4])
try:
    for _ in range(300):
        if p.poll() is not None: sys.exit("server exited")
        try: socket.create_connection(("127.0.0.1", 18233), timeout=1).close(); break
        except OSError: time.sleep(1)
    py(); go()  # warm-up both
    for blk in range(2):
        for name in ("py", "go", "go", "py"):
            r, ev, rep = (py if name == "py" else go)()
            print(f"block {blk+1} {name} rate {r:.2f} tok/s events {ev} reported {rep}", flush=True)
finally:
    p.send_signal(signal.SIGTERM); p.wait(timeout=60); log.close()
