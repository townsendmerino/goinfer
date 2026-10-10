#!/usr/bin/env python3
"""G-S10q-c's reference (docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE"): a layer-streaming Hugging
Face float32 forward of Qwen3-VL MoE on image prompts, step (b')'s design (scripts/g31b_hf_stream.py) for this family.

The 30B is about 120 GB in float32, beyond this box's 62 GB, so this runs Hugging Face's OWN modules LAYER-MAJOR: every
sequence goes through decoder layer 0 (its weights read from the shards, the experts converted as transformers' own
conversion converts them, cast to float32, used, dropped), then layer 1, and so on, keeping only the [tokens, hidden]
states between layers. The image features come from HF's own vision tower (built alone from the config and the shards'
model.visual.*), on HF's own pixel values; the m-RoPE positions from HF's own get_rope_index. What this script owns is
the loop order, the splice of the image rows into the embeddings, the DeepStack adds after layers 0..2 (HF's
_deepstack_process arithmetic), the final norm and the untied head; no layer arithmetic is re-implemented.

Modes:  stream    layer-major (the 30B; the tiny for control 1)
        ordinary  the model as Hugging Face runs it, whole (the tiny's control reference; the dense Qwen3-VL-2B sibling)
Defects (stream only; the registered controls, each must break control 1's bar on the tiny):
        nodeepstack (the DeepStack adds skipped), swap (the first two layers visited in swapped order), norenorm (the
        router's top-k weights not renormalised).

Input: a JSON list of {"ids": [...], "path": [...], "image": <path>}. The sequence is ids + path[:-1] and the logits are
taken at positions len(ids)-1+j, j in 0..len(path)-1 (the logits that predict path[j]). ids must already hold the
image's pad tokens, as many as HF's processor makes merged rows (checked).
Output: <out>.f32, float32 [sum(len(path)), vocab] in sequence order, and <out>.json with the offsets.
Usage: ~/.venv-vl/bin/python -I scripts/q3vlmoe_hf_stream.py <model dir> <seqs.json> <out prefix> [stream|ordinary] [defect]
"""
import json
import os
import sys
import time
import types

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

T0 = time.time()


def hb(m):
    print(f"[q3vlmoe {time.time() - T0:6.0f}s] {m}", flush=True)


class Shards:
    def __init__(self, d):
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.map = json.load(open(idx))["weight_map"]
        else:
            with safe_open(os.path.join(d, "model.safetensors"), "pt") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        self.d, self.open = d, {}

    def get(self, key):
        fn = self.map[key]
        if fn not in self.open:
            self.open[fn] = safe_open(os.path.join(self.d, fn), "pt")
        return self.open[fn].get_tensor(key).float()

    def under(self, prefix):
        return {k[len(prefix):]: self.get(k) for k in self.map if k.startswith(prefix)}


def classes(model_type):
    if model_type == "qwen3_vl_moe":
        import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as M
        return M, M.Qwen3VLMoeVisionModel, M.Qwen3VLMoeTextDecoderLayer, M.Qwen3VLMoeTextRotaryEmbedding, M.Qwen3VLMoeTextRMSNorm, M.Qwen3VLMoeModel
    import transformers.models.qwen3_vl.modeling_qwen3_vl as M
    return M, M.Qwen3VLVisionModel, M.Qwen3VLTextDecoderLayer, M.Qwen3VLTextRotaryEmbedding, M.Qwen3VLTextRMSNorm, M.Qwen3VLModel


def image_inputs(proc, path):
    enc = proc(images=[Image.open(path).convert("RGB")], return_tensors="pt")
    return enc["pixel_values"].float(), enc["image_grid_thw"]


def rope_index(Model, cfg, ids, grid):
    """HF's own get_rope_index on a stub: it reads only the config and get_vision_position_ids, no weights."""
    stub = types.SimpleNamespace(config=cfg)
    stub.get_vision_position_ids = lambda *a, **k: Model.get_vision_position_ids(stub, *a, **k)
    mm = (ids == cfg.image_token_id).int()
    pos, _ = Model.get_rope_index(stub, ids, mm, image_grid_thw=grid)
    return pos


def stream(model_dir, seqs, defect):
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.masking_utils import create_causal_mask
    cfg = AutoConfig.from_pretrained(model_dir)
    M, Tower, Layer, Rotary, RMSNorm, Model = classes(cfg.model_type)
    tc = cfg.text_config
    tc._attn_implementation = "sdpa"
    # A bare config names no experts implementation, and a standalone decoder layer then does not run what from_pretrained
    # runs (G-S10q-c's control 1 read 1.26 relative before this). Name from_pretrained's choice; HF's three implementations
    # agree to 1.3e-6 on the tiny.
    tc._experts_implementation = "grouped_mm"
    cfg.vision_config._attn_implementation = "sdpa"
    proc = AutoImageProcessor.from_pretrained(model_dir)
    sh = Shards(model_dir)
    # the tower, alone
    vis = Tower._from_config(cfg.vision_config, dtype=torch.float32)
    missing, unexpected = vis.load_state_dict(sh.under("model.visual."), strict=False)
    missing = [k for k in missing if not k.endswith("inv_freq")]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    vis.eval()
    H, L = tc.hidden_size, tc.num_hidden_layers
    emb = sh.get("model.language_model.embed_tokens.weight")
    norm = RMSNorm(H, eps=tc.rms_norm_eps)
    norm.weight.data = sh.get("model.language_model.norm.weight")
    head = sh.get("lm_head.weight") if "lm_head.weight" in sh.map else emb
    rot = Rotary(tc)
    S = []
    for s in seqs:
        ids = torch.tensor([s["ids"] + s["path"][:-1]])
        pv, grid = image_inputs(proc, s["image"])
        with torch.no_grad():
            out = vis(pv, grid_thw=grid)
        merged, deep = out.pooler_output, list(out.deepstack_features)
        mask = ids == cfg.image_token_id
        if int(mask.sum()) != merged.shape[0]:
            sys.exit(f"{s['image']}: {int(mask.sum())} image tokens in the ids, the tower gives {merged.shape[0]} rows")
        x = emb[ids[0]].clone()[None]
        x[mask] = merged
        pos = rope_index(Model, cfg, ids, grid)
        pe = rot(x, pos)
        # HF's text model, given three-row m-RoPE positions, builds the mask and calls the layers with position_ids None
        # (text_position_ids is taken only from a four-row tensor). Passing the temporal row instead made create_causal_mask
        # read the image span's repeated positions as packed sequences: G-S10q-c's control 1 read 1.26 relative until this.
        am = create_causal_mask(config=tc, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=None)
        S.append(dict(x=x, mask=mask, deep=deep, pe=pe, am=am, tpos=None))
    if os.environ.get("Q3VLMOE_DUMP"):
        torch.save(S[0]["x"].clone(), os.path.join(os.environ["Q3VLMOE_DUMP"], "embed.pt"))
    hb(f"tower and embeddings: {len(seqs)} sequences")
    order = list(range(L))
    if defect == "swap":
        order[0], order[1] = order[1], order[0]
    if defect == "norenorm":  # the registered router defect: the top-k weights left as the softmax gave them
        def fwd(self, h):
            h = h.reshape(-1, self.hidden_dim)
            logits = torch.nn.functional.linear(h, self.weight)
            probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
            top, idx = torch.topk(probs, self.top_k, dim=-1)
            return logits, top.to(logits.dtype), idx
        M.Qwen3VLMoeTextTopKRouter.forward = fwd
    for step, i in enumerate(order):
        t = time.time()
        with torch.device("meta"):
            layer = Layer(tc, i)
        layer.to_empty(device="cpu")
        sd = sh.under(f"model.language_model.layers.{i}.")
        want = dict(layer.state_dict())
        for k in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):  # transformers' own conversion: Transpose(1, 2, check_dims=True)
            if k in sd and tuple(sd[k].shape) != tuple(want[k].shape):
                sd[k] = sd[k].transpose(1, 2).contiguous()
        layer.load_state_dict(sd, strict=True)
        nb = {n for n, _ in layer.named_buffers()}
        assert nb <= set(want), f"layer {i}: non-persistent buffers {nb - set(want)} would be uninitialised after to_empty"
        layer.eval()
        lt = time.time() - t
        t = time.time()
        with torch.no_grad():
            for st in S:
                h = layer(st["x"], attention_mask=st["am"], position_ids=st["tpos"], past_key_values=None, position_embeddings=st["pe"])
                h = h[0] if isinstance(h, tuple) else h
                if defect != "nodeepstack" and step < len(st["deep"]):  # HF adds set l after the l-th layer visited
                    h = h.clone()
                    h[st["mask"], :] = h[st["mask"], :] + st["deep"][step]
                st["x"] = h
        if os.environ.get("Q3VLMOE_DUMP"):  # debugging: each step's state of the first sequence
            torch.save(S[0]["x"].clone(), os.path.join(os.environ["Q3VLMOE_DUMP"], f"step{step}.pt"))
        del layer
        hb(f"layer {step + 1}/{L} (index {i}): load {lt:.1f}s, {len(seqs)} sequences {time.time() - t:.1f}s")
    out = []
    for st, s in zip(S, seqs):
        h = norm(st["x"][0, len(s["ids"]) - 1: len(s["ids"]) - 1 + len(s["path"])])
        out.append((h @ head.T).detach().numpy())
    return out


def ordinary(model_dir, seqs):
    from transformers import AutoConfig, AutoImageProcessor
    cfg = AutoConfig.from_pretrained(model_dir)
    if cfg.model_type == "qwen3_vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration as C
    else:
        from transformers import Qwen3VLForConditionalGeneration as C
    model = C.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="sdpa").eval()
    proc = AutoImageProcessor.from_pretrained(model_dir)
    out = []
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor([s["ids"] + s["path"][:-1]])
            pv, grid = image_inputs(proc, s["image"])
            mm = (ids == cfg.image_token_id).int()
            lg = model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, mm_token_type_ids=mm).logits[0]
            out.append(lg[len(s["ids"]) - 1: len(s["ids"]) - 1 + len(s["path"])].float().numpy())
    return out


if __name__ == "__main__":
    model_dir, seqs_path, outp = sys.argv[1:4]
    mode = sys.argv[4] if len(sys.argv) > 4 else "stream"
    defect = sys.argv[5] if len(sys.argv) > 5 else "none"
    if model_dir.startswith("/Volumes/") or model_dir.startswith("/srv/models"):
        sys.exit(f"{model_dir} is on the archive (CLAUDE.md)")
    seqs = json.load(open(seqs_path))
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    hb(f"{mode} {defect}: {len(seqs)} sequences, {sum(len(s['path']) for s in seqs)} positions")
    res = stream(model_dir, seqs, "" if defect == "none" else defect) if mode == "stream" else ordinary(model_dir, seqs)
    flat = np.concatenate(res).astype("<f4")
    flat.tofile(outp + ".f32")
    json.dump({"mode": mode, "defect": defect, "positions": [len(s["path"]) for s in seqs], "vocab": int(flat.shape[1])}, open(outp + ".json", "w"))
    hb(f"wrote {outp}.f32 {flat.shape}")
