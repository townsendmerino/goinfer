#!/usr/bin/env python3
"""Step (b') of the Gemma 4 31B work (docs/tasks/task-multimodal-support-2026-10.md, "Gemma 4 31B, step (b')", registered before this code): a layer-streaming Hugging Face float32 reference.
The 31B is 59 GB in bf16, so an ordinary float32 forward (about 124 GB) does not fit this box's 62 GB. This runs Hugging Face's OWN decoder-layer modules LAYER-MAJOR: every sequence goes through layer 0
(its weights read from the safetensors shards, cast to float32, used, dropped), then layer 1, and so on, keeping only the [tokens, hidden] states between layers. What this script owns is the loop order,
the scaled embedding, the masks (HF's own helpers), the final norm, the tied head and the softcap; no layer arithmetic is re-implemented.

Modes:  stream    layer-major, one layer's weights resident at a time (the 31B and the controls)
        ordinary  Gemma4ForCausalLM float32 as Hugging Face runs it (the control's reference; fits only for small models)
Defects (stream only; the registered planted-defect controls, each must break control 1's bar): nosoftcap, nonorm, swap (the first two layers visited in swapped order), fullmask (sliding layers get a full
causal mask), kvtie (global layers get their own random v_proj: the K=V tie dropped), noscale (embedding not scaled by sqrt(hidden)).

Input: a JSON list of {"ids": [...], "path": [...]}. The sequence is ids + path[:-1] and the logits are taken at positions len(ids)-1+j, j in 0..len(path)-1 (the logits that predict path[j]).
Output: <out>.f32, float32 [sum(len(path)), vocab] in sequence order, and <out>.json with the offsets.
Usage: ~/g4venv/bin/python -I scripts/g31b_hf_stream.py <model dir> <seqs.json> <out prefix> [stream|ordinary] [defect]"""
import json, os, sys, time
from collections import UserDict
import numpy as np
import torch
from safetensors import safe_open

T0 = time.time()
def hb(m): print(f"[g31b {time.time() - T0:6.0f}s] {m}", flush=True)

class Shards:
    def __init__(self, d):
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            f = safe_open(os.path.join(d, "model.safetensors"), "pt"); self.map = {k: "model.safetensors" for k in f.keys()}
        self.d = d; self.open = {}
        cands = [k for k in self.map if k.endswith("layers.0.self_attn.q_proj.weight")]
        assert len(cands) == 1, cands
        self.prefix = cands[0][: -len("layers.0.self_attn.q_proj.weight")]
    def get(self, key):
        fn = self.map[key]
        if fn not in self.open: self.open[fn] = safe_open(os.path.join(self.d, fn), "pt")
        return self.open[fn].get_tensor(key).float()
    def layer(self, i):
        p = f"{self.prefix}layers.{i}."
        return {k[len(p):]: self.get(k) for k in self.map if k.startswith(p)}

def text_config(d):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(d)
    return getattr(c, "text_config", None) or c

def build(model_dir, seqs, defect):
    import transformers.models.gemma4.modeling_gemma4 as G
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    cfg = text_config(model_dir); sh = Shards(model_dir)
    # A config that never went through a PreTrainedModel has no attention implementation: the mask helpers then return no mask (sdpa's is_causal convention) while a standalone layer runs the eager
    # kernel, which would attend to the future. Both must name the SAME implementation, the one from_pretrained gives an ordinary run.
    cfg._attn_implementation = "sdpa"
    assert not cfg.hidden_size_per_layer_input and not getattr(cfg, "num_kv_shared_layers", 0) and not cfg.enable_moe_block, "this streamer is for the dense, PLE-free, unshared 31B shape"
    H, L = cfg.hidden_size, cfg.num_hidden_layers
    emb = G.Gemma4TextScaledWordEmbedding(cfg.vocab_size, H, cfg.pad_token_id, embed_scale=(1.0 if defect == "noscale" else H ** 0.5))
    emb.weight.data = sh.get(sh.prefix + "embed_tokens.weight")
    norm = G.Gemma4RMSNorm(H, eps=cfg.rms_norm_eps); norm.weight.data = sh.get(sh.prefix + "norm.weight")
    rot = G.Gemma4TextRotaryEmbedding(cfg)
    types = set(cfg.layer_types)
    states, pos, pe, masks = [], [], [], []
    for s in seqs:
        ids = torch.tensor([s["ids"] + s["path"][:-1]])
        x = emb(ids).detach(); p = torch.arange(ids.shape[1])[None]
        mk = dict(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=p)
        m = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
        if defect == "fullmask": m["sliding_attention"] = m["full_attention"]
        states.append(x); pos.append(p); masks.append(m); pe.append({t: rot(x, p, t) for t in types})
    order = list(range(L))
    if defect == "swap": order[0], order[1] = order[1], order[0]
    return dict(G=G, cfg=cfg, sh=sh, emb=emb, norm=norm, states=states, pos=pos, pe=pe, masks=masks, order=order, H=H, L=L)

def load_layer(S, i, defect):
    G, cfg = S["G"], S["cfg"]
    with torch.device("meta"):
        layer = G.Gemma4TextDecoderLayer(cfg, i)
    layer.to_empty(device="cpu")
    sd = S["sh"].layer(i)
    if defect == "kvtie" and layer.self_attn.v_proj is None:  # the tie dropped: the global layer gets a V projection of its own (seeded)
        g = torch.Generator().manual_seed(1000 + i); w = torch.randn(layer.self_attn.k_proj.weight.shape, generator=g) * 0.02
        layer.self_attn.v_proj = torch.nn.Linear(w.shape[1], w.shape[0], bias=False); layer.self_attn.v_proj.weight.data = w
        sd["self_attn.v_proj.weight"] = w
    missing, unexpected = layer.load_state_dict(sd, strict=True), None
    nb = {n for n, _ in layer.named_buffers()}; pers = set(layer.state_dict().keys())
    assert nb <= pers, f"layer {i}: non-persistent buffers {nb - pers} would be uninitialised memory after to_empty"
    return layer

def stream(model_dir, seqs, defect):
    S = build(model_dir, seqs, defect); cfg = S["cfg"]
    for step, i in enumerate(S["order"]):
        t = time.time(); layer = load_layer(S, i, defect); lt = time.time() - t
        t = time.time()
        with torch.no_grad():
            for k in range(len(seqs)):
                S["states"][k] = layer(S["states"][k], None, shared_kv_states=UserDict(), position_embeddings=S["pe"][k][cfg.layer_types[i]],
                                       attention_mask=S["masks"][k][cfg.layer_types[i]], position_ids=S["pos"][k], past_key_values=None)
        del layer
        hb(f"layer {step + 1}/{S['L']} (index {i}, {cfg.layer_types[i]}): load {lt:.1f}s, {len(seqs)} sequences {time.time() - t:.1f}s")
    out = []
    W = S["emb"].weight  # tied head
    for k, s in enumerate(seqs):
        h = S["states"][k][0, len(s["ids"]) - 1: len(s["ids"]) - 1 + len(s["path"])]
        if defect != "nonorm": h = S["norm"](h)
        lg = h @ W.T
        if defect != "nosoftcap" and cfg.final_logit_softcapping is not None:
            lg = torch.tanh(lg / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
        out.append(lg.detach().numpy())
    return out

def ordinary(model_dir, seqs):
    from transformers import AutoModelForCausalLM  # maps a text-only gemma4_text to Gemma4ForCausalLM and the multimodal E4B/31B configs to Gemma4ForConditionalGeneration
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32); model.eval()
    out = []
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor([s["ids"] + s["path"][:-1]])
            lg = model(input_ids=ids).logits[0]
            out.append(lg[len(s["ids"]) - 1: len(s["ids"]) - 1 + len(s["path"])].float().numpy())
    return out

if __name__ == "__main__":
    model_dir, seqs_path, outp = sys.argv[1:4]
    mode = sys.argv[4] if len(sys.argv) > 4 else "stream"; defect = sys.argv[5] if len(sys.argv) > 5 else "none"
    seqs = json.load(open(seqs_path))
    hb(f"{mode} {defect}: {len(seqs)} sequences, {sum(len(s['path']) for s in seqs)} positions")
    res = stream(model_dir, seqs, "" if defect == "none" else defect) if mode == "stream" else ordinary(model_dir, seqs)
    flat = np.concatenate(res).astype("<f4"); flat.tofile(outp + ".f32")
    json.dump({"mode": mode, "defect": defect, "positions": [len(s["path"]) for s in seqs], "vocab": int(flat.shape[1])}, open(outp + ".json", "w"))
    hb(f"wrote {outp}.f32 {flat.shape}")
