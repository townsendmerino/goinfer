import json, os, signal, socket, subprocess, sys, time, urllib.request
OUT = sys.argv[1]
prompts = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts/prompts.json")))
text = prompts["G20:128"]["text"]
env = dict(os.environ, OLLAMA_MODELS=os.path.expanduser("~/ollama-0325/models"), OLLAMA_HOST="127.0.0.1:11555")
env.pop("LD_LIBRARY_PATH", None)
log = open(os.path.join(OUT, "ollama.log"), "wb")
p = subprocess.Popen([os.path.expanduser("~/ollama-0325/bin/ollama"), "serve"], env=env, stdout=log, stderr=log)
try:
    for _ in range(60):
        try: socket.create_connection(("127.0.0.1", 11555), timeout=1).close(); break
        except OSError: time.sleep(1)
    body = {"model": "g20", "stream": True, "messages": [{"role": "user", "content": text}],
            "options": {"num_predict": 64, "num_ctx": 4096, "num_gpu": 0, "temperature": 0, "seed": 1}}
    req = urllib.request.Request("http://127.0.0.1:11555/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    ev = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if line: ev.append(json.loads(line))
    done = [e for e in ev if e.get("done")]
    tok = [e for e in ev if not e.get("done")]
    keys = {}
    for e in tok:
        for k, v in e.get("message", {}).items():
            if v: keys[k] = keys.get(k, 0) + 1
    print(json.dumps({"events": len(ev), "token_events": len(tok), "eval_count": done[0].get("eval_count") if done else None,
                      "message_keys": keys, "first": [e.get("message") for e in tok[:3]]}, indent=1))
finally:
    p.send_signal(signal.SIGTERM); p.wait(timeout=60); log.close()
