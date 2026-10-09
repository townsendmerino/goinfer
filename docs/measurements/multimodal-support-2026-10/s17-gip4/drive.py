import base64, json, math, os, subprocess, sys, time, urllib.request
out = sys.argv[1]
imgs = ["gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png", "glm_ocr/table.png"]
arms = {"bridge-74830779": os.path.expanduser("~/goinfer-bench/s7-2026-10-09/serve-metal"),
        "resident-80be9921": os.path.expanduser("~/goinfer-bench/s17-gip4/serve-metal")}
res = {}
for lab, binp in arms.items():
    log = open(f"{out}/{lab}.log", "w")
    p = subprocess.Popen([binp, "--model", os.path.expanduser("~/models/gemma-3-4b-it"), "--backend", "metal", "--addr", "127.0.0.1:18613"], stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:18613/v1/models", timeout=2); break
            except Exception:
                time.sleep(1)
        for im in imgs:
            b64 = base64.b64encode(open("testdata/" + im, "rb").read()).decode()
            body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                    {"type": "text", "text": "What does this image show? Answer briefly."}]}], "max_tokens": 32, "temperature": 0, "logprobs": True, "top_logprobs": 3}
            t = time.time()
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18613/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
            c = r["choices"][0]
            res[(lab, im)] = (c["message"]["content"], c["logprobs"]["content"])
            print(f"{lab} {im}: {time.time()-t:.1f}s {c['message']['content'][:90]!r}", flush=True)
    finally:
        p.terminate(); p.wait(); time.sleep(5)
json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(f"{out}/replies.json", "w"))
for im in imgs:
    (ra, rd), (oa, od) = res[("bridge-74830779", im)], res[("resident-80be9921", im)]
    if ra == oa:
        print(f"{im}: IDENTICAL"); continue
    i = 0
    while i < min(len(rd), len(od)) and rd[i]["token"] == od[i]["token"]:
        i += 1
    if i >= min(len(rd), len(od)):
        print(f"{im}: differ in length only"); continue
    top = rd[i]["top_logprobs"]; pt = math.exp(top[0]["logprob"])
    po = [math.exp(x["logprob"]) for x in top if x["token"] == od[i]["token"]]
    print(f"{im}: first difference at token {i}: bridge {rd[i]['token']!r}, resident {od[i]['token']!r}; bridge top-3 " +
          ", ".join(f"{x['token']!r} {math.exp(x['logprob']):.3f}" for x in top) + f"; near-tie {bool(po and po[0] >= pt/2)}")
