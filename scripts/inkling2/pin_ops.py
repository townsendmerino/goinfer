# Usage: python3 scripts/inkling2/pin_ops.py internal/runtime/inkling2/testdata/ops_golden.json
# Needs a transformers build with models/inkling2_loop (main at 8e094f8970 or later) and torch.
# Writes per-op tier-0 vectors for goinfer's internal/runtime/inkling2 kernels, computed by transformers' own
# Inkling2Loop modules (modeling_inkling2_loop.py) on small random inputs.
import json, sys, math, torch
import torch.nn.functional as F
from transformers.models.inkling2_loop import modeling_inkling2_loop as M
from transformers import Inkling2LoopTextConfig
torch.manual_seed(7)
cfg = Inkling2LoopTextConfig(hidden_size=8, loop_gate_rank=3, log_scaling_n_floor=4, log_scaling_alpha=0.3,
                             rms_norm_eps=1e-6, d_rel=4, conv_kernel_size=4, num_hidden_layers=6, layer_types=["hybrid_sliding"]*6)
cfg._attn_implementation = "eager"
L = lambda t: t.detach().reshape(-1).tolist()
out = {}
with torch.no_grad():
    # clip: 3 heads x 5, threshold 1.5, learned scale
    c = M.Inkling2LoopScaledL2Clip(3, 5, 1.5); c.scale.copy_(1 + 0.5 * torch.randn(3, 5))
    x = torch.randn(1, 1, 3, 5) * torch.tensor([0.2, 1.0, 2.0]).view(1, 1, 3, 1)
    out["clip"] = {"x": L(x), "scale": L(c.scale), "d": 5, "thr": 1.5, "y": L(c(x))}
    # log scale
    ls = M.Inkling2LoopLogScale(cfg)
    pos = torch.arange(12).view(1, 12)
    one = torch.ones(1, 12, 1, 1)
    out["logscale"] = {"floor": 4, "alpha": 0.3, "y": L(ls(one, pos))}
    # depth mix: 3 queries over 2 blocks + current
    m = M.Inkling2LoopBlockAttentionResidual(cfg, num_queries=3); m.query.copy_(torch.randn(3, 8))
    blocks = torch.randn(1, 1, 2, 8) * 3; cur = torch.randn(1, 1, 8) * 2
    ys = m(blocks, cur)
    out["mix"] = {"queries": L(m.query), "states": L(torch.cat([blocks, cur.unsqueeze(-2)], -2)), "nq": 3, "y": L(torch.stack(ys, -2))}
    # controller
    g = M.Inkling2LoopMessageGate(cfg)
    for p in g.parameters(): p.copy_(0.5 * torch.randn(p.shape))
    xh = torch.randn(1, 1, 8) * 2; msg = torch.randn(1, 1, 8)
    out["control"] = {"x": L(xh), "m": L(msg), "down": L(g.down_proj.weight), "gain": L(g.message_gate_proj.weight),
                      "forget": L(g.forget_gate_proj.weight), "rank": 3, "y": L(g(xh, msg))}
    # static conv over 6 positions, 8 channels, kernel 4
    sc = M.Inkling2LoopShortConvolution(8, 4, layer_idx=0, conv_idx=0).float()
    sc.conv1d.weight.copy_(torch.randn(8, 1, 4) * 0.5)
    xs = torch.randn(1, 6, 8)
    out["conv"] = {"x": L(xs), "taps": L(sc.conv1d.weight.squeeze(1)), "k": 4, "n": 6, "y": L(sc(xs))}
    # dynamic conv: 8 channels in 2 heads, input 8
    dc = M.Inkling2LoopDynamicShortConvolution(8, 8, 2, 4, layer_idx=0, conv_idx=0)
    dc.static_weight.copy_(torch.randn(4, 8) * 0.5); dc.weight_proj.weight.copy_(torch.randn(8, 8) * 0.5)
    xd = torch.randn(1, 6, 8); dyn_in = torch.randn(1, 6, 8)
    dyn = dc.weight_proj(dyn_in)  # [1,6, heads*k] with tap index = lag
    # goinfer's static taps layout is llama.cpp's: [channels, k], tap k-1 the current token
    taps = dc.static_weight.flip(0).T.contiguous()
    out["dynconv"] = {"x": L(xd), "taps": L(taps), "dyn": L(dyn), "k": 4, "n": 6, "y": L(dc(xd, dyn_in))}
    # situ
    s = M.Inkling2LoopSoftcappedSwiGLU(Inkling2LoopTextConfig(mlp_gate_softcap=4.0, mlp_up_softcap=25.0))
    ga = torch.randn(10) * 6; up = torch.randn(10) * 40
    out["situ"] = {"gate": L(ga), "up": L(up), "bg": 4.0, "bu": 25.0, "y": L(s(ga, up))}
    # attention with relative bias and per-token sinks: 4 heads, 2 kv heads, hd 3, 7 positions, window 4, extent 5
    H, KV, hd, dr, T, W, E = 4, 2, 3, 4, 7, 4, 5
    q = torch.randn(1, H, T, hd); k = torch.randn(1, KV, T, hd); v = torch.randn(1, KV, T, hd)
    rel = torch.randn(1, T, H, dr); proj = torch.randn(dr, E); sinks = torch.randn(1, H, T, 1)
    dist = torch.arange(T)[:, None] - torch.arange(T)[None, :]
    rel_logits = (rel @ proj).transpose(1, 2)
    gi = dist.clamp(0, E - 1).expand(1, H, -1, -1)
    bias = rel_logits.gather(-1, gi).masked_fill((dist < 0) | (dist >= E), 0.0)
    mask = torch.where((dist >= 0) & (dist < W), 0.0, float("-inf")).view(1, 1, T, T)
    class Mod(torch.nn.Module):
        num_key_value_groups = H // KV
        training = False
    ctx, _ = M.eager_attention_forward(Mod(), q, k, v, mask, scaling=hd ** -0.5, position_bias=bias, sinks=sinks)
    out["attend"] = {"heads": H, "kv": KV, "hd": hd, "drel": dr, "n": T, "window": W, "extent": E, "scale": hd ** -0.5,
                     "q": L(q.transpose(1, 2)), "k": L(k.transpose(1, 2)), "v": L(v.transpose(1, 2)),
                     "rel": L(rel), "relbias": L(proj.T.contiguous()), "sinks": L(sinks.squeeze(-1).transpose(1, 2)),
                     "y": L(ctx)}
json.dump(out, open(sys.argv[1], "w"))
print("ok", list(out))
