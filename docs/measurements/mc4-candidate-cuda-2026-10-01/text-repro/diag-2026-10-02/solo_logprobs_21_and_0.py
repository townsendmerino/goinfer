import os, sys, json
sys.path.insert(0, "/home/francis/mycode/goinfer/scripts")
import bench_w7_plain as w7, bench_spec_copy as sc
secs = sc.sections("cc5f8c2c", 3500, 30)
with w7.GoinferServer("cuda", "", "/dev/null") as srv:
    for idx in (21, 0):
        body = dict(sc.request(secs[idx]), max_tokens=8, logprobs=True, top_logprobs=4)
        lat, resp = w7.post(srv.url, body, timeout=300)
        lp = resp["choices"][0]["logprobs"]["content"]
        print(f"--- section {idx}: {resp['choices'][0]['message']['content'][:30]!r}")
        for i, t in enumerate(lp):
            top = [(x["token"], round(x["logprob"], 3)) for x in t["top_logprobs"]]
            gap = round(top[0][1] - top[1][1], 3) if len(top) > 1 else None
            print(f"  tok {i}: {t['token']!r:14} gap top1-top2 = {gap}   {top[:3]}")
