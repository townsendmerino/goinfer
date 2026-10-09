#!/usr/bin/env python3
"""The 896-pixel investigation's Hugging Face half (docs/tasks/task-multimodal-support-2026-10.md, "The 896-pixel image prefill"): per-layer differencing of goinfer's two prefills against
Hugging Face's float32 text model, fed the SAME inputs. cuda/s896_layerdump_test.go writes, per image, the ids, the m-RoPE positions, the merged image rows, the DeepStack sets and each arm's
[layers][rows*hidden] residual (the residual after layer l and its DeepStack add: HF's hidden_states[l+1]). This script runs Qwen3VLTextModel on inputs_embeds (embedding of the ids with the image
rows replaced by the merged rows, no scale), the same position_ids, visual_pos_masks and deepstack_visual_embeds, and prints per layer: cosine and relative error of each arm against HF, image rows and
text rows apart. Layer N-1 is read from a pre-hook on the final norm (hidden_states[N] is post-norm: CLAUDE.md's differencing caution).
Usage: ~/g4venv/bin/python -I scripts/s896_hf_layers.py <model dir> <dump dir> <tag> [out json] [logits dir: the G-S10j-style dump whose <tag>.off/on.f32 holds each arm's own step-0 logits]"""
import json, sys, time
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration

model_dir, dump, tag = sys.argv[1:4]
out_json = sys.argv[4] if len(sys.argv) > 4 else f"{dump}/{tag}.layers.json"
logit_dir = sys.argv[5] if len(sys.argv) > 5 else None
T0 = time.time()
def hb(m): print(f"[s896 hf {time.time() - T0:5.0f}s] {m}", flush=True)
meta = json.load(open(f"{dump}/{tag}.meta.json")); n, h, L, s, ni = meta["n"], meta["hidden"], meta["layers"], meta["start"], meta["nImg"]
feats = np.fromfile(f"{dump}/{tag}.feats.f32", dtype="<f4").reshape(ni, h)
deep = np.fromfile(f"{dump}/{tag}.deep.f32", dtype="<f4").reshape(meta["nDeep"], ni, h)
arms = {a: np.fromfile(f"{dump}/{tag}.{a}.f32", dtype="<f4").reshape(L, n, h) for a in ("cpu", "cuda")}
hb("loading float32 model")
model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32); model.eval()
lm = model.model.language_model
assert len(lm.layers) == L, (len(lm.layers), L)
ids = torch.tensor([meta["ids"]])
emb = lm.embed_tokens(ids).detach().clone()
emb[0, s:s + ni] = torch.from_numpy(feats)
pos = torch.tensor(meta["mrope"], dtype=torch.long).T[:, None, :].contiguous()  # [3, 1, n]
vmask = torch.zeros(1, n, dtype=torch.bool); vmask[0, s:s + ni] = True
ds = [torch.from_numpy(deep[i]) for i in range(meta["nDeep"])]
# The state ENTERING each layer l+1 (and the final norm) is the residual after layer l AND its DeepStack add, which the parent loop applies after the layer returns. Pre-hooks see exactly that;
# output_hidden_states in recent transformers records layer OUTPUTS through hooks (no DeepStack add), so it is only used below to say which convention it follows.
entering = {}
def mk(i):
    def f(mod, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        entering[i] = x.detach().clone()
    return f
hooks = [lm.layers[i].register_forward_pre_hook(mk(i), with_kwargs=True) for i in range(1, L)]
hooks.append(lm.norm.register_forward_pre_hook(lambda mod, args: entering.__setitem__(L, args[0].detach().clone())))
hb("forward")
with torch.no_grad():
    out = lm(inputs_embeds=emb, position_ids=pos, visual_pos_masks=vmask, deepstack_visual_embeds=ds, output_hidden_states=True, use_cache=False)
for hk in hooks: hk.remove()
ref = [entering[l + 1][0].numpy() for l in range(L)]
hs = [x[0].numpy() for x in out.hidden_states]
conv = []
for l in (0, 1, 2, 3):
    a = hs[l + 1]; b = ref[l]
    conv.append(f"layer {l}: hidden_states[{l + 1}] vs state-entering-next: max|diff| {np.abs(a - b).max():.3g} (image rows {np.abs(a[s:s+ni] - b[s:s+ni]).max():.3g})")
hb("output_hidden_states convention: " + "; ".join(conv))
# HF's own rope index against goinfer's (an independent check of the positions we fed)
try:
    ri, _ = model.model.get_rope_index(ids, torch.tensor([meta["grid"]]), None, torch.ones(1, n, dtype=torch.long))
    hb(f"HF get_rope_index equals goinfer's mropePos: {bool((ri[:, 0, :].T.numpy() == np.array(meta['mrope'])).all())}")
except Exception as e:
    hb(f"get_rope_index check skipped: {e}")
img = np.zeros(n, bool); img[s:s + ni] = True
def stats(a, b, sel):
    a, b = a[sel], b[sel]
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    r = np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(b, axis=-1) + 1e-30)
    return float(c.min()), float(c.mean()), float(r.mean())
rows = []
print("layer | image rows: cos(cpu,HF) min/mean rel | cos(cuda,HF) min/mean rel || text rows: cpu min/mean rel | cuda min/mean rel")
for l in range(L):
    rec = {"layer": l}
    for nm, sel in (("img", img), ("txt", ~img)):
        for a in ("cpu", "cuda"):
            rec[f"{nm}_{a}"] = stats(arms[a][l], ref[l], sel)
    rows.append(rec)
    f = lambda t: f"{t[0]:.4f} {t[1]:.4f} {t[2]:.3f}"
    print(f"{l:2d}    | {f(rec['img_cpu'])} | {f(rec['img_cuda'])} || {f(rec['txt_cpu'])} | {f(rec['txt_cuda'])}")
# The last row, layer by layer: it is the one whose logits are the first token.
print("\nlast row (the first token's position), cosine / relative error against HF:")
for l in (0, 4, 8, 12, 16, 20, 24, 26, 27):
    a = [ (float((arms[x][l][-1] * ref[l][-1]).sum() / (np.linalg.norm(arms[x][l][-1]) * np.linalg.norm(ref[l][-1]))), float(np.linalg.norm(arms[x][l][-1] - ref[l][-1]) / np.linalg.norm(ref[l][-1]))) for x in ("cpu", "cuda")]
    print(f"  layer {l:2d}: cpu {a[0][0]:.5f} {a[0][1]:.4f} | cuda {a[1][0]:.5f} {a[1][1]:.4f}")
# Per row: where does the CUDA arm lose to the CPU arm? delta = cos(cpu,HF) - cos(cuda,HF) (positive: CUDA farther), at a middle and the last layer; the text rows after the image one by one, the image block by position bucket.
print("\nper-row delta = cos(cpu,HF) - cos(cuda,HF)   (positive: the CUDA arm is farther from HF)")
def rowcos(a, b): return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
for l in (8, 16, 24, 27):
    d = rowcos(arms["cpu"][l], ref[l]) - rowcos(arms["cuda"][l], ref[l])
    buckets = [s + int(k * ni / 8) for k in range(9)]
    img_b = " ".join(f"{d[buckets[k]:buckets[k + 1]].mean():+.4f}" for k in range(8))
    pre = d[:s].mean()
    post = d[s + ni:]
    print(f"  layer {l:2d}: before-image rows {pre:+.4f} | image block in 8 buckets: {img_b} | after-image rows (n={len(post)}): " + " ".join(f"{x:+.3f}" for x in post))
# Body against head. HF's logits from its own last residual; each arm's "body" logits are the EXACT float32 final norm + head applied to that arm's own last residual row (so only the body's error is in
# them); each arm's actual step-0 logits (what the goinfer head path produced) come from the dump when given. Cosine and KL(HF || x) over the softmax.
head = model.lm_head
def logits_of(res):
    with torch.no_grad():
        return head(lm.norm(torch.from_numpy(np.ascontiguousarray(res))[None, :]))[0].numpy().astype(np.float64)
def cosv(a, b): return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
def kl(p_logits, q_logits):
    p = np.exp(p_logits - p_logits.max()); p /= p.sum(); q = np.exp(q_logits - q_logits.max()); q /= q.sum()
    return float((p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum())
Lhf = logits_of(ref[L - 1][-1])
res = {"hf": Lhf}
for a in ("cpu", "cuda"):
    res[f"{a}_body"] = logits_of(arms[a][L - 1][-1])
if logit_dir:
    V = len(Lhf)
    for a, f in (("cpu", "off"), ("cuda", "on")):
        res[f"{a}_actual"] = np.fromfile(f"{logit_dir}/{tag}.{f}.f32", dtype="<f4").reshape(-1, V)[0].astype(np.float64)
print("\nstep-0 logits against HF (argmax; cosine; KL(HF||x) nats):")
for k, v in res.items():
    if k == "hf": print(f"  hf            argmax {int(v.argmax())}"); continue
    print(f"  {k:12s}  argmax {int(v.argmax())}  cos {cosv(Lhf, v):.5f}  KL {kl(Lhf, v):.4f}")
json.dump({"tag": tag, "rows": rows, "step0": {k: {"argmax": int(v.argmax()), "cos": cosv(Lhf, v), "kl": kl(Lhf, v)} for k, v in res.items() if k != "hf"}}, open(out_json, "w"))
hb("done")
