import json, sys, numpy as np, torch
from transformers import AutoModelForCausalLM
name, ckpt = sys.argv[1], sys.argv[2]
D = f"/tmp/claude-1000/-home-francis-mycode-goinfer/8cb247f8-a43d-46b5-8a98-1abefe2d63b4/scratchpad/dump/{name}"
meta = json.load(open(D + "/meta.json"))
V = meta["vocab"]
torch.set_num_threads(16)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
def load(f, n): return np.fromfile(f"{D}/{f}.f32", dtype=np.float32).reshape(n, V).astype(np.float64)
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
out = {}
def hf(ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0].to(torch.float64).numpy()
for key, ids in (("golden", meta["golden"]), ("long", meta["long"])):
    h = hf(ids); c = load(key + "_cpu", len(ids)); r = load(key + "_res", len(ids))
    rows = []
    for i in range(len(ids)):
        rows.append(dict(pos=i, res_hf=cos(r[i], h[i]), cpu_hf=cos(c[i], h[i]), res_cpu=cos(r[i], c[i]),
                         relres_hf=rel(r[i], h[i]), relcpu_hf=rel(c[i], h[i]), relres_cpu=rel(r[i], c[i])))
    out[key] = rows
# continuation steps: HF over prompt + the CPU-greedy tokens; step k logits are at position len(prompt)-1+k
ids = meta["golden"] + meta["cont_tokens"]
h = hf(ids); c = load("cont_cpu", 8); r = load("cont_res", 8)
n0 = len(meta["golden"])
steps = []
for k in range(8):
    hk = h[n0 - 1 + k]
    top = lambda x: np.argsort(-x)[:2]
    t_h, t_c, t_r = top(hk), top(c[k]), top(r[k])
    rng = lambda x: float(x.max() - x.min())
    steps.append(dict(step=k, res_hf=cos(r[k], hk), cpu_hf=cos(c[k], hk), res_cpu=cos(r[k], c[k]),
        hf_top2=[int(t_h[0]), int(t_h[1])], cpu_top2=[int(t_c[0]), int(t_c[1])], res_top2=[int(t_r[0]), int(t_r[1])],
        hf_gap_pct_of_range=100 * float(hk[t_h[0]] - hk[t_h[1]]) / rng(hk),
        cpu_gap_pct=100 * float(c[k][t_c[0]] - c[k][t_c[1]]) / rng(c[k]),
        res_gap_pct=100 * float(r[k][t_r[0]] - r[k][t_r[1]]) / rng(r[k])))
out["cont"] = steps
json.dump(out, open(D + "/hf_ref.json", "w"), indent=1)
for key in ("golden", "long"):
    rows = out[key]
    print(key, "worst res~HF %.6f (pos %d) | worst CPU~HF %.6f (pos %d) | worst res~CPU %.6f (pos %d)" % (
        min(r["res_hf"] for r in rows), min(rows, key=lambda r: r["res_hf"])["pos"],
        min(r["cpu_hf"] for r in rows), min(rows, key=lambda r: r["cpu_hf"])["pos"],
        min(r["res_cpu"] for r in rows), min(rows, key=lambda r: r["res_cpu"])["pos"]))
