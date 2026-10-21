#!/usr/bin/env python3
"""KV0-H A2 item 5: the Claude Code context rule, applied. The largest context up to 49,152, in steps of 4,096, that leaves at least 512 MiB of the card free
after the first Claude Code request. For each candidate, descending: start the study server at that --ctx (f32 KV: the shipped precision of qwen3_5 on CUDA,
cuda/kvpolicy.go:cudaKVF16Decline), read the free VRAM after the load and, when that is enough, replay the pilot's recorded first Claude Code request (max_tokens
cut to 8: its prompt is unchanged, only the reply is shortened) and read the free VRAM again. Writes calibration.json and prints the chosen context.

  calibrate_ctx.py <serve binary> <model> <first-request body file> <ids key file> <out dir> [--min-free 512] [--max 49152] [--step 4096]
"""
import json, os, signal, subprocess, sys, time, urllib.request

serve, model, bodyf, key, out = sys.argv[1:6]
MIN_FREE = 512
CANDS = list(range(49152, 24575, -4096))
PORT = 18091
os.makedirs(out, exist_ok=True)


def free_mib():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


rec = json.load(open(bodyf))
body = json.loads(rec["body"])
body["max_tokens"] = 8
body["stream"] = False
data = json.dumps(body).encode()
results = []
chosen = None
for ctx in CANDS:
    r = dict(ctx=ctx)
    log = open(os.path.join(out, f"calib-serve-{ctx}.log"), "w")
    base = free_mib()
    p = subprocess.Popen([serve, "--model", model, "--backend", "cuda", "--ctx", str(ctx), "--addr", f"127.0.0.1:{PORT}", "--kv-sessions", "1", "--ids-key-file", key, "--log-requests"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time(); up = False
    while time.time() - t0 < 180 and p.poll() is None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1).read(); up = True; break
        except Exception:
            time.sleep(0.5)
    banner = open(os.path.join(out, f"calib-serve-{ctx}.log")).read()
    resident = "decode path: cuda-resident" in banner
    r.update(server_up=up, resident=resident, free_before_mib=base)
    if up and resident:
        time.sleep(1)
        r["free_after_load_mib"] = free_mib()
        if r["free_after_load_mib"] >= MIN_FREE:
            t1 = time.time()
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}{rec['path'].split('?')[0]}", data=data, headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                resp = json.load(urllib.request.urlopen(req, timeout=900))
                r["first_request_s"] = round(time.time() - t1, 1)
                r["prompt_tokens"] = (resp.get("usage") or {}).get("input_tokens")
            except Exception as e:
                r["first_request_error"] = f"{type(e).__name__}: {e}"
            time.sleep(1)
            r["free_after_first_request_mib"] = free_mib()
    for ln in banner.splitlines():
        if "KV f" in ln or "fit:" in ln:
            r.setdefault("banner", []).append(ln.strip()[:200])
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=60)
    except Exception:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    for _ in range(30):
        if free_mib() >= base - 64:
            break
        time.sleep(1)
    ok = r.get("free_after_first_request_mib", -1) >= MIN_FREE and "first_request_error" not in r
    r["passes"] = ok
    results.append(r)
    say(f"ctx {ctx}: resident={resident} free after load {r.get('free_after_load_mib')} MiB, after first request {r.get('free_after_first_request_mib')} MiB -> {'PASS' if ok else 'no'}")
    if ok:
        chosen = ctx
        break
json.dump(dict(rule="largest ctx <= 49152 (steps of 4096) leaving >= 512 MiB free after the first Claude Code request", min_free_mib=MIN_FREE, chosen=chosen, results=results),
          open(os.path.join(out, "calibration.json"), "w"), indent=1)
print(chosen if chosen else "NONE")
sys.exit(0 if chosen else 2)
