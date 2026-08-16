#!/usr/bin/env python
"""Reference-side acceptance for the DFlash drafter — the ATTRIBUTION instrument for P10
kill-gate 2, and the reason the gate-2 miss did not become a wrong verdict.

    ~/.venv-vl/bin/python scripts/ref_dflash_accept.py

Measured 2026-08-15 (linux-62gb, 160 new tokens/prompt): code 6.37, math 3.86, chat 2.31
tok/verify — versus goinfer's 2.90 / 2.79 / 2.13 at int8. The 6.37 reproduces upstream's
published ~6.0 on code, which is what establishes that the claims transfer and the gap is
ours. See docs/spec/08-dspark-dflash.md.

Original docstring: reference-side acceptance for attribution: is the sub-3.0 number OUR setup or the
claim not transferring? Runs z-lab's own spec_generate loop (copied verbatim from the
checkpoint's MIT dflash.py so the measurement is theirs, with only an acceptance counter
added) at THEIR precision (bf16) and THEIR chat template, on our suite."""
import sys, types, torch, os
_s = types.ModuleType("datasets")
for n in ("load_dataset","Features","Sequence","Value"): setattr(_s,n,None)
sys.modules["datasets"] = _s
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

D=os.path.expanduser("~/models/qwen3-4b-dflash"); T=os.path.expanduser("~/models/qwen3-4b")
SUITES={
 "code":["Write a Python function that returns the nth Fibonacci number.",
         "Write a Go function that reverses a slice of ints in place.",
         "Write a SQL query that selects the top 5 customers by total order value."],
 "math":["What is 17 * 23? Show your working.",
         "A train travels 120 km in 1.5 hours. What is its average speed in km/h?"],
 "chat":["Explain what a hash table is, in two sentences.",
         "Give me three tips for keeping houseplants alive."]}
MAXNEW=160

def run(drf, tgt, ids, max_new_tokens):
    """z-lab dflash.py::spec_generate, verbatim apart from returning acceptance_lengths."""
    from transformers import DynamicCache
    sample = sys.modules[drf.__class__.__module__].sample
    num_input_tokens = ids.shape[1]; max_length = num_input_tokens + max_new_tokens
    block_size = drf.block_size
    output_ids = torch.full((1, max_length + block_size), drf.mask_token_id, dtype=torch.long)
    position_ids = torch.arange(output_ids.shape[1]).unsqueeze(0)
    pkv_t, pkv_d = DynamicCache(), DynamicCache()
    out = tgt(ids, position_ids=position_ids[:, :num_input_tokens], past_key_values=pkv_t,
              use_cache=True, logits_to_keep=1, output_hidden_states=True)
    output_ids[:, :num_input_tokens] = ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(out.logits, 0.0)
    from importlib import import_module
    ecf = sys.modules[drf.__class__.__module__].extract_context_feature
    target_hidden = ecf(out.hidden_states, drf.target_layer_ids)
    acc=[]; start = ids.shape[1]
    while start < max_length:
        blk = output_ids[:, start:start+block_size].clone()
        noise = tgt.model.embed_tokens(blk)
        dl = tgt.lm_head(drf(target_hidden=target_hidden, noise_embedding=noise,
                             position_ids=position_ids[:, pkv_d.get_seq_length(): start+block_size],
                             past_key_values=pkv_d, use_cache=True, is_causal=False)[:, -block_size+1:, :])
        pkv_d.crop(start)
        blk[:, 1:] = sample(dl)
        out = tgt(blk, position_ids=position_ids[:, start:start+block_size],
                  past_key_values=pkv_t, use_cache=True, output_hidden_states=True)
        post = sample(out.logits, 0.0)
        a = (blk[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start:start+a+1] = blk[:, :a+1]
        output_ids[:, start+a+1] = post[:, a]
        start += a + 1
        pkv_t.crop(start)
        target_hidden = ecf(out.hidden_states, drf.target_layer_ids)[:, :a+1, :]
        acc.append(a+1)
    return acc

tok=AutoTokenizer.from_pretrained(T)
tgt=AutoModelForCausalLM.from_pretrained(T, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
drf=AutoModel.from_pretrained(D, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).eval()
with torch.no_grad():
    for suite, prompts in SUITES.items():
        rounds=toks=0
        for p in prompts:
            txt=tok.apply_chat_template([{"role":"user","content":p}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
            ids=tok(txt, return_tensors="pt").input_ids
            a=run(drf,tgt,ids,MAXNEW); rounds+=len(a); toks+=sum(a)
        print(f"REF {suite:5s} {rounds:4d} rounds {toks:4d} tokens => {toks/rounds:.2f} tok/verify", flush=True)
