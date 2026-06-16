#!/usr/bin/env python
"""Tiny-random Nemotron-H (NemotronHForCausalLM) checkpoint + text golden — the
Nemotron-H parity fixture. Nemotron-H is a SINGLE-OP-per-block hybrid: each layer is
one of {mamba, attention, mlp}, not mixer+FFN. Mamba-2 reuses Granite's conventions;
the mlp blocks are non-gated relu^2. Exercises all three block kinds + the head.

    ~/.venv-vl/bin/python scripts/pin_nemotron_tiny.py
    -> testdata/nemotron_tiny_text_golden.json + testdata/nemotron-tiny/
"""
import json, os, torch
from transformers import NemotronHConfig, NemotronHForCausalLM
HERE=os.path.dirname(__file__)
OUT=os.path.join(HERE,"..","testdata","nemotron_tiny_text_golden.json")
CKPT=os.path.join(HERE,"..","testdata","nemotron-tiny")
CFG=dict(vocab_size=256, hidden_size=64, num_hidden_layers=5,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    mamba_num_heads=8, mamba_head_dim=8, ssm_state_size=16, n_groups=1, conv_kernel=4,
    intermediate_size=128, layers_block_type=['mamba','attention','mlp','mamba','attention'],
    layer_norm_epsilon=1e-5, tie_word_embeddings=False)
PROMPT=[2,7,42,100,5,200,13,88]; N_NEW=6
def main():
    torch.manual_seed(0)
    c=NemotronHConfig(**CFG)
    print("=== config of interest ===")
    cd=c.to_dict()
    for k in ['rope_theta','attention_bias','mlp_hidden_act','mamba_hidden_act','use_bias','rope_scaling','use_conv_bias','attn_implementation','position_embedding_type']:
        print(f"  {k} = {cd.get(k,'<absent>')}")
    m=NemotronHForCausalLM(c).eval().to(torch.float32)
    with torch.no_grad():
        ids=torch.tensor([PROMPT])
        last=m(input_ids=ids,use_cache=False).logits[0,-1].float().tolist()
        cur,cont=list(PROMPT),[]
        for _ in range(N_NEW):
            o=m(input_ids=torch.tensor([cur]),use_cache=False)
            cont.append(int(o.logits[0,-1].argmax())); cur.append(cont[-1])
    g=dict(note="tiny NemotronH text fwd fp32", config=CFG, prompt_ids=PROMPT,
        argmax=int(torch.tensor(last).argmax()), last_logits=last, n_new=N_NEW, continuation_ids=cont)
    os.makedirs(os.path.dirname(OUT),exist_ok=True); json.dump(g,open(OUT,"w"))
    print(f"argmax={g['argmax']} cont={cont}")
    m.save_pretrained(CKPT,safe_serialization=True); print("saved",CKPT)
if __name__=="__main__": main()
