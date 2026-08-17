#!/usr/bin/env python
"""Tiny-random Nemotron 3 Nano (NemotronHForCausalLM, MoE variant) checkpoint + text
golden. Sibling of pin_nemotron_tiny.py — same single-op-per-block hybrid, but the
pattern's "E" character (transformers' own _pattern_to_list maps E -> "moe",
configuration_nemotron_h.py) makes some layers a sigmoid-routed, non-gated relu² MoE
FFN instead of the dense relu² mlp — the primitive goinfer's nemotron_h adapter added
for Nemotron 3 Nano (decoder/forward_nemotron.go's nemotronMoE/nemotronExpertFFN).
Exercises mamba + attention + moe (the real Nemotron 3 Nano pattern has no plain "-"
dense-mlp layers at all, confirmed against the real checkpoint's config.json, so this
fixture doesn't either — faithful to the real family, not just to what's convenient).

    ~/.venv-nemotron3/bin/python scripts/pin_nemotron3nano_tiny.py
    -> testdata/nemotron3nano_tiny_text_golden.json + testdata/nemotron3nano-tiny/
"""
import json, os, torch
from transformers import NemotronHConfig, NemotronHForCausalLM
HERE=os.path.dirname(__file__)
OUT=os.path.join(HERE,"..","testdata","nemotron3nano_tiny_text_golden.json")
CKPT=os.path.join(HERE,"..","testdata","nemotron3nano-tiny")
CFG=dict(vocab_size=256, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    mamba_num_heads=8, mamba_head_dim=8, ssm_state_size=16, n_groups=1, conv_kernel=4,
    intermediate_size=128,
    # layers_block_type DIRECTLY, goinfer's own bare vocabulary (mamba/attention/moe/mlp) — NOT
    # hybrid_override_pattern. transformers' own __post_init__ expands a pattern string through
    # ITS internal _pattern_to_list vocabulary ("linear_attention"/"full_attention", not
    # "mamba"/"attention") and SAVES THAT to the checkpoint's config.json — a real checkpoint
    # ships hybrid_override_pattern directly (goinfer's own normalizeNemotronBlocks parses that
    # string itself, independent of HF's internal naming), but a config with layers_block_type
    # ALREADY populated skips that parse entirely and goinfer's switch wouldn't recognize
    # "linear_attention"/"full_attention". Setting the list directly here, exactly like the
    # existing pin_nemotron_tiny.py already does, sidesteps the mismatch rather than papering
    # over it with an alias goinfer doesn't need for the real checkpoint.
    layers_block_type=['mamba','moe','mamba','attention','mamba','moe'],
    n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
    n_shared_experts=1, moe_shared_expert_intermediate_size=64,  # deliberately 2x, not 1x — matches the real checkpoint's ratio
    norm_topk_prob=True, routed_scaling_factor=2.5, n_group=1, topk_group=1,
    layer_norm_epsilon=1e-5, tie_word_embeddings=False)
PROMPT=[2,7,42,100,5,200,13,88]; N_NEW=6
def main():
    torch.manual_seed(0)
    c=NemotronHConfig(**CFG)
    print("=== config of interest ===")
    cd=c.to_dict()
    for k in ['layer_types','n_routed_experts','num_experts_per_tok','moe_intermediate_size',
              'moe_shared_expert_intermediate_size','n_shared_experts','norm_topk_prob',
              'routed_scaling_factor','n_group','topk_group','mlp_hidden_act','mamba_hidden_act']:
        print(f"  {k} = {cd.get(k,'<absent>')}")
    m=NemotronHForCausalLM(c).eval().to(torch.float32)
    with torch.no_grad():
        ids=torch.tensor([PROMPT])
        last=m(input_ids=ids,use_cache=False).logits[0,-1].float().tolist()
        cur,cont=list(PROMPT),[]
        for _ in range(N_NEW):
            o=m(input_ids=torch.tensor([cur]),use_cache=False)
            cont.append(int(o.logits[0,-1].argmax())); cur.append(cont[-1])
    g=dict(note="tiny NemotronH-MoE (Nemotron 3 Nano) text fwd fp32", config=CFG, prompt_ids=PROMPT,
        argmax=int(torch.tensor(last).argmax()), last_logits=last, n_new=N_NEW, continuation_ids=cont)
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(g,open(OUT,"w"))
    print(f"argmax={g['argmax']} cont={cont}")
    m.save_pretrained(CKPT,safe_serialization=True); print("saved",CKPT)
if __name__=="__main__": main()
