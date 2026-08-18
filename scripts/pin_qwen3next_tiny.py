#!/usr/bin/env python
"""Tiny-random Qwen3-Next (Qwen3NextForCausalLM) checkpoint + text golden — the
Qwen3-Next parity fixture. Qwen3-Next is the same Gated-DeltaNet/softmax/MoE
hybrid shape as qwen3_5_moe, but the real released config computes the per-layer
pattern (full_attention_interval) instead of stating it (layer_types), and
carries partial_rotary_factor as a top-level field with no rope_parameters
object at all — this fixture deliberately uses that same flat shape (no
layer_types, no rope_parameters) so the goinfer loader is exercised on the real
config's actual shape, not a convenient one.

    ~/.venv-nemotron3/bin/python scripts/pin_qwen3next_tiny.py
    -> testdata/qwen3next_tiny_text_golden.json + testdata/qwen3next-tiny/
"""
import json, os, torch
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
HERE=os.path.dirname(__file__)
OUT=os.path.join(HERE,"..","testdata","qwen3next_tiny_text_golden.json")
CKPT=os.path.join(HERE,"..","testdata","qwen3next-tiny")
CFG=dict(vocab_size=256, hidden_size=64, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    intermediate_size=32, full_attention_interval=4,
    linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=4,
    num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
    shared_expert_intermediate_size=16, norm_topk_prob=True,
    partial_rotary_factor=0.25, rope_theta=1000000.0, rms_norm_eps=1e-6,
    tie_word_embeddings=False)
PROMPT=[2,7,42,100,5,200,13,88]; N_NEW=6
def main():
    torch.manual_seed(0)
    c=Qwen3NextConfig(**CFG)
    print("=== config of interest ===")
    cd=c.to_dict()
    for k in ['layer_types','full_attention_interval','partial_rotary_factor','rope_scaling','rope_parameters','hidden_act']:
        print(f"  {k} = {cd.get(k,'<absent>')}")
    m=Qwen3NextForCausalLM(c).eval().to(torch.float32)
    with torch.no_grad():
        ids=torch.tensor([PROMPT])
        last=m(input_ids=ids,use_cache=False).logits[0,-1].float().tolist()
        cur,cont=list(PROMPT),[]
        for _ in range(N_NEW):
            o=m(input_ids=torch.tensor([cur]),use_cache=False)
            cont.append(int(o.logits[0,-1].argmax())); cur.append(cont[-1])
    g=dict(note="tiny Qwen3-Next text fwd fp32", config=CFG, prompt_ids=PROMPT,
        argmax=int(torch.tensor(last).argmax()), last_logits=last, n_new=N_NEW, continuation_ids=cont)
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(g,open(OUT,"w"))
    print(f"argmax={g['argmax']} cont={cont}")
    m.save_pretrained(CKPT,safe_serialization=True); print("saved",CKPT)

    # transformers' own save_pretrained EXPANDS full_attention_interval into an explicit
    # layer_types list and rope_theta/partial_rotary_factor into a nested rope_parameters
    # object — even though flat/computed values were passed to the constructor. The REAL
    # released Qwen3-Next config has NEITHER (verified against the actual HF repo): only
    # full_attention_interval and a top-level rope_theta. Rewrite the saved config back to
    # that real shape (same underlying values, different JSON shape) so this fixture
    # actually exercises normalizeQwen3NextLayerTypes + the flat-RoPE path — the two real
    # deltas from qwen3_5_moe this family exists to test — rather than the
    # already-populated shape qwen3_5_moe's own fixture already covers.
    cfgpath = os.path.join(CKPT, "config.json")
    saved = json.load(open(cfgpath))
    assert saved.pop("layer_types", None) == ["linear_attention"]*3 + ["full_attention"] + ["linear_attention"]*2, \
        "layer_types didn't match the expected full_attention_interval=4 pattern — formula assumption stale?"
    rp = saved.pop("rope_parameters")
    assert rp["rope_theta"] == CFG["rope_theta"] and rp["partial_rotary_factor"] == CFG["partial_rotary_factor"], \
        "rope_parameters values didn't match what was passed in — can't safely drop them"
    saved["full_attention_interval"] = CFG["full_attention_interval"]
    saved["rope_theta"] = CFG["rope_theta"]
    saved["rope_scaling"] = None
    json.dump(saved, open(cfgpath, "w"), indent=2, sort_keys=True)
    print("rewrote config.json to the real release's flat shape (layer_types/rope_parameters -> full_attention_interval/rope_theta)")
if __name__=="__main__": main()
