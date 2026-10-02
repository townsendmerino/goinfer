import sys, json
sys.path.insert(0, "/home/francis/mycode/goinfer/scripts")
import bench_w7_plain as w7, bench_spec_copy as sc
secs = sc.sections("cc5f8c2c", 3500, 30)
def ask(srv, k, n=6):
    body = dict(sc.request(secs[k]), max_tokens=n, logprobs=True, top_logprobs=3)
    _, r = w7.post(srv.url, body, timeout=300)
    lp = r["choices"][0]["logprobs"]["content"]
    return [(t["token"], t["logprob"]) for t in lp], r["usage"].get("prefill_reused_tokens"), r["usage"]["prompt_tokens"]
res = {}
for label, pre in (("21 first", []), ("21 after 0", [0]), ("21 after 9", [9])):
    with w7.GoinferServer("cuda", "", "/dev/null") as srv:
        w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4, "temperature": 0})
        for k in pre: ask(srv, k)
        res[label] = ask(srv, 21)
        print(f"{label:11s} prompt_tokens={res[label][2]} reused={res[label][1]}  logprobs " + " ".join(f"{t[1]:.4f}" for t in res[label][0]), flush=True)
base = res["21 first"][0]
for label in ("21 after 0", "21 after 9"):
    print(label, "max |dlogprob| vs first:", [round(abs(a[1] - b[1]), 4) for a, b in zip(base, res[label][0])])
