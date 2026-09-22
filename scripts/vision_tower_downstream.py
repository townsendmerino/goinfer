#!/usr/bin/env python3
"""R8 gate 4 driver (docs/measurements/vision-tower-downstream-PREREGISTERED.md): serve gemma-3-4b-it on CUDA under each vision-attention arm,
send real images with greedy + logprobs, and compare. Usage: vision_tower_downstream.py out.json [--serve PATH] [--model DIR]"""
import argparse, base64, io, json, os, random, signal, socket, subprocess, sys, time, urllib.request
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
IMAGES = [os.path.join(ROOT, "testdata/gemma3_preprocess_image.png"), "/usr/share/gutenprint/samples/profile.jpg",
          "/usr/share/inkscape/screens/start-welcome.png", "/usr/share/plasma/avatars/Office Worker Konqi.png",
          "/usr/share/wallpapers/nobara-wallpaper-2026.png", "/usr/share/iso-flag-png/_earth_pernefeldt.png",
          "/usr/share/iso-flag-png/ad.png", "/usr/share/iso-flag-png/au.png"]
PORT = 8097
def png_b64(img):
    b = io.BytesIO(); img.save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()
def jitter(img, seed):
    a = np.asarray(img.convert("RGB")).astype(np.int16); rng = np.random.default_rng(seed)
    m = rng.random(a.shape) < 0.05; d = rng.choice([-1, 1], size=a.shape)
    return Image.fromarray(np.clip(a + m * d, 0, 255).astype(np.uint8))
def wait_port(t=300):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            socket.create_connection(("127.0.0.1", PORT), 1).close(); return True
        except OSError: time.sleep(0.5)
    return False
def ask(b64):
    body = {"model": "bench", "stream": False, "max_tokens": 64, "temperature": 0, "logprobs": True, "top_logprobs": 5,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this image in detail."},
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t0
    c = r["choices"][0]; lp = (c.get("logprobs") or {}).get("content") or []
    return {"text": c["message"]["content"], "tokens": [x["token"] for x in lp], "logprob0": lp[0]["logprob"] if lp else None,
            "top0": [(t["token"], t["logprob"]) for t in (lp[0]["top_logprobs"] if lp else [])], "prompt_tokens": r["usage"]["prompt_tokens"], "seconds": dt}
def run_arm(serve, model, arm, jobs):
    env = dict(os.environ, GOINFER_CUDA_VISION_ATTN=arm)
    p = subprocess.Popen([serve, "-model", f"bench={model}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    try:
        if not wait_port(): raise RuntimeError("serve did not come up")
        out = {}
        for k, b64 in jobs:
            out[k] = ask(b64); print(f"[{arm}] {k}: {out[k]['seconds']:.1f}s  {len(out[k]['text'].split())} words  {out[k]['text'][:70]!r}", flush=True)
        return out
    finally:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM); time.sleep(3)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--serve", default=os.path.expanduser("~/bench-cur/serve-cuda-0921"))
    ap.add_argument("--model", default=os.path.expanduser("~/models/gemma-3-4b-it")); a = ap.parse_args()
    imgs = [Image.open(p) for p in IMAGES]
    plain = [(f"img{i}", png_b64(im.convert("RGB"))) for i, im in enumerate(imgs)]
    pert = [(f"img{i}", png_b64(jitter(im, 1000 + i))) for i, im in enumerate(imgs)]
    res = {"images": IMAGES}
    res["N"] = run_arm(a.serve, a.model, "bm128", plain)
    res["E"] = run_arm(a.serve, a.model, "exact", plain + [(k + "#2", b) for k, b in plain[:3]])   # E2: a repeat of the first 3
    res["P"] = run_arm(a.serve, a.model, "exact", pert)
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)
main()
