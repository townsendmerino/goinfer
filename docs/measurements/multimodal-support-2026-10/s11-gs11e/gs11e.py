"""G-S11e (S11, docs/tasks/task-multimodal-support-2026-10.md): served two-image requests on one real checkpoint.
Arms: new binary Metal two images; new binary CPU two images; new binary Metal one image; old (pre-S11) binary Metal one
image. 32 greedy tokens, top-3 logprobs. Run from the repo root."""
import base64, json, math, os, subprocess, sys, time, urllib.request
out, name = sys.argv[1], sys.argv[2]
model_args = sys.argv[3:]
NEW = os.path.expanduser("~/goinfer-bench/s11/serve-metal")
OLD = os.path.expanduser("~/goinfer-bench/s17-gip4/serve-metal")
def img(p):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open("testdata/" + p, "rb").read()).decode()}}
TWO = [{"type": "text", "text": "Here are two images. "}, img("glm_ocr/table.png"), {"type": "text", "text": " and "}, img("glm_ocr/formula.png"),
       {"type": "text", "text": " What does each image show? Answer briefly, one line per image."}]
ONE = [img("glm_ocr/table.png"), {"type": "text", "text": "What does this image show? Answer briefly."}]
# The two-image comparison keeps the tower on the CPU in both arms (G-S3c's method), so only the decoder's backend differs.
arms = [("new-metal-two", NEW, ["--backend", "metal", "-vision-device", "cpu"], TWO), ("new-cpu-two", NEW, ["--backend", "cpu"], TWO),
        ("new-metal-one", NEW, ["--backend", "metal"], ONE), ("old-metal-one", OLD, ["--backend", "metal"], ONE)]
res = {}
for lab, binp, flags, content in arms:
    log = open(f"{out}/{name}-{lab}.log", "w")
    p = subprocess.Popen([binp, *model_args, *flags, "--addr", "127.0.0.1:18620"], stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen("http://127.0.0.1:18620/v1/models", timeout=2); break
            except Exception:
                if p.poll() is not None:
                    raise SystemExit(f"{lab}: serve exited, see {out}/{name}-{lab}.log")
                time.sleep(1)
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": 96, "temperature": 0, "logprobs": True, "top_logprobs": 3}
        t = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18620/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
        except urllib.error.HTTPError as e:
            print(f"{lab}: HTTP {e.code}: {e.read().decode()[:300]}", flush=True); continue
        c = r["choices"][0]
        res[lab] = (c["message"]["content"], c["logprobs"]["content"], r["usage"]["prompt_tokens"])
        print(f"{lab} ({time.time()-t:.1f}s, {r['usage']['prompt_tokens']} prompt tokens): {c['message']['content']!r}", flush=True)
    finally:
        p.terminate(); p.wait(); time.sleep(5)
json.dump(res, open(f"{out}/{name}-replies.json", "w"))
def cmp(a, b, n=32):  # the first n generated tokens
    (_, la, _), (_, lb, _) = res[a], res[b]
    la, lb = la[:n], lb[:n]
    if [x["token"] for x in la] == [x["token"] for x in lb]:
        return f"IDENTICAL (first {n} tokens)"
    i = 0
    while i < min(len(la), len(lb)) and la[i]["token"] == lb[i]["token"]:
        i += 1
    if i >= min(len(la), len(lb)):
        return "differ in length only"
    top = la[i]["top_logprobs"]; pt = math.exp(top[0]["logprob"])
    po = [math.exp(x["logprob"]) for x in top if x["token"] == lb[i]["token"]]
    return f"first difference at token {i}: {a} {la[i]['token']!r}, {b} {lb[i]['token']!r}; {a} top-3 " + ", ".join(f"{x['token']!r} {math.exp(x['logprob']):.3f}" for x in top) + f"; R10 near-tie {bool(po and po[0] >= pt/2)}"
if "new-metal-two" in res:
    t = res["new-metal-two"][0].lower()
    print(f"{name}: names both: table {'table' in t}, formula {any(w in t for w in ('integral', 'gaussian', 'formula', 'equation'))}")
if "new-metal-two" in res and "new-cpu-two" in res:
    print(f"{name}: two images, Metal against the CPU (reference): {cmp('new-cpu-two', 'new-metal-two')}")
if "new-metal-one" in res and "old-metal-one" in res:
    print(f"{name}: one image, new binary against the pre-S11 one: {cmp('old-metal-one', 'new-metal-one')}")
