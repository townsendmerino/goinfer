# Usage: python3 scripts/inkling2/pin_tiny.py <out dir>, then scripts/inkling2/convert_tiny.py <out dir> <out dir>/model.gguf;
# decoder/testdata/inkling2-loop-tiny holds model.gguf and golden.json gzipped. Needs transformers main (8e094f8970 or later).
# Builds a random-initialized tiny Inkling2LoopForCausalLM, saves it (HF format), and writes the golden:
# per-visit decoder-layer outputs and the logits of a 12-token prompt (full-sequence forward), plus a
# cached incremental forward to check HF's own cache path agrees.
import json, gzip, sys, os, math, torch
from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig
torch.manual_seed(1234)
out_dir = sys.argv[1]
cfg = Inkling2LoopTextConfig(
    vocab_size=96, unpadded_vocab_size=90, hidden_size=64, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
    sliding_window_size=4, d_rel=4, rel_extent=8, log_scaling_n_floor=4, log_scaling_alpha=0.5,
    layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
    n_layers_per_attn_res_block=2, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
    loop_layer_index_start=2, loop_layer_index_end=4, loop_num_passes=3, loop_gate_rank=8,
    qk_l2_clip_threshold=6.0, rel_l2_clip_threshold=3.0, mlp_gate_softcap=2.0, mlp_up_softcap=3.0,
    final_logit_softcapping=5.0, conv_kernel_size=4, route_scale=8.0, rms_norm_eps=1e-6,
    eos_token_id=89, pad_token_id=None,
)
cfg._attn_implementation = "eager"
model = Inkling2LoopForCausalLM(cfg).float().eval()
g = torch.Generator().manual_seed(99)
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("norm.weight") or "layernorm" in name:
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        elif name.endswith(".scale"):
            p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=g))
        elif "query" in name:
            p.copy_(0.3 * torch.randn(p.shape, generator=g))
        elif "e_score_correction_bias" in name:
            p.copy_(0.2 * torch.randn(p.shape, generator=g))
        else:
            fan = p.shape[-1] if p.ndim >= 2 else 1
            std = 0.5 if p.ndim < 2 else 1.0 / math.sqrt(fan)
            p.copy_(std * torch.randn(p.shape, generator=g))
    for name, b in model.named_buffers():
        if "e_score_correction_bias" in name:
            b.copy_(0.2 * torch.randn(b.shape, generator=g))
ids = torch.tensor([[3, 17, 42, 5, 88, 23, 23, 61, 7, 70, 11, 54]])
visits = []
names = []
def hook(mod, args, kwargs, out):
    visits.append((out[0] if isinstance(out, tuple) else out)[0].detach().clone())
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(hook, with_kwargs=True)
with torch.no_grad():
    full = model(input_ids=ids, use_cache=False).logits[0]
nv = len(visits)
assert nv == 10, nv
hid = [v.tolist() for v in visits]
visits.clear()
# cached incremental: prefill 5, then one token at a time
with torch.no_grad():
    o = model(input_ids=ids[:, :5], use_cache=True)
    cache = o.cache if hasattr(o, "cache") else o.past_key_values
    rows = [o.logits[0, i] for i in range(5)]
    for t in range(5, ids.shape[1]):
        o = model(input_ids=ids[:, t:t+1], cache=cache, use_cache=True)
        cache = o.cache if hasattr(o, "cache") else o.past_key_values
        rows.append(o.logits[0, 0])
inc = torch.stack(rows)
print("cached-vs-full max|diff|", (inc - full).abs().max().item())
model.save_pretrained(out_dir, safe_serialization=True)
golden = {"input_ids": ids[0].tolist(), "visits": hid, "logits": full.tolist(),
          "note": "visits[v][t] is decoder-layer output at visit v (10 visits: 0,1, core 2,3 x3, 4,5), position t"}
with open(os.path.join(out_dir, "golden.json"), "w") as f:
    json.dump(golden, f)
print("logits argmax", full.argmax(-1).tolist())
