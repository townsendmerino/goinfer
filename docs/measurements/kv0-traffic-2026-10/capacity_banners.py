#!/usr/bin/env python3
"""KV0 traffic study, the bar's capacity arm: start serve on the 8 GB card with each flag set, read the context and KV plan it
prints, stop it. Nothing is generated.

  capacity_banners.py <serve binary> <model path> <label> [-- flags ...] [-- flags ...]

Each `--`-separated flag group is one load; the banner lines "context:", "KV plan:" and "fit:" and the VRAM used after the load are printed.
"""
import json, os, re, signal, subprocess, sys, time, urllib.request

serve, model, label = sys.argv[1:4]
groups, cur = [], None
for a in sys.argv[4:]:
    if a == "--":
        cur = []; groups.append(cur)
    elif cur is not None:
        cur.append(a)
for g in groups or [[]]:
    log = f"/tmp/kv0-capacity-{os.getpid()}.log"
    p = subprocess.Popen([serve, "--model", model, "--backend", "cuda", "--addr", "127.0.0.1:18191", *g], stdout=open(log, "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, "GOINFER_SWAP_GUARD": "4096"})
    ok = False
    for _ in range(240):
        if p.poll() is not None:
            break
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:18191/v1/models", timeout=2)).get("data"):
                ok = True; break
        except Exception:
            time.sleep(1)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    txt = open(log).read()
    print(f"== {label} {' '.join(g) or '(defaults)'}: {'up' if ok else 'DID NOT COME UP'}, VRAM used {smi}")
    for line in txt.splitlines():
        if re.search(r"context:|KV plan|fit:|cuda: resident context|refus|declin|does not fit|decode path|resident build", line):
            print("   ", line.strip()[:260])
    try:
        os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=60)
    except Exception:
        os.killpg(p.pid, signal.SIGKILL)
    time.sleep(4)
