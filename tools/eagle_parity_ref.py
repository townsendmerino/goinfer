# EAGLE-3 head parity reference (05, docs/spec). Run in a torch venv with transformers
# (e.g. ~/.venv-vl). Loads Qwen3-1.7B (HF) + the AngelSlim EAGLE-3 head .bin, runs the
# head's step-0 forward in torch, and scores per-position acceptance α = 1-TV(p,q)
# against the target. Used to confirm goinfer's Go head forward is faithful and to
# find the protocol: d2t = offset (target = i + d2t[i]); feature norm (hidden_norm) is
# essential; the head needs the CHAT regime (α 0.34 raw -> 0.53 chat). The Go forward
# matches this reference at step 0.
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
base_dir="/home/francis/models/qwen3-1.7b-bf16"; head_bin="/home/francis/models/qwen3-1.7b-eagle3/pytorch_model.bin"
tok=AutoTokenizer.from_pretrained(base_dir); model=AutoModelForCausalLM.from_pretrained(base_dir,torch_dtype=torch.float32).eval()
raw=torch.load(head_bin,map_location="cpu",weights_only=True); sd={k:v.float() for k,v in raw.items() if v.dtype!=torch.bool}; d2t=raw["d2t"].long()
def rms(x,w,eps=1e-6): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*w
ew=model.get_input_embeddings().weight
def step0(h3,tid):
    feat=h3@sd["fc.weight"].T
    e=rms(ew[tid],sd["midlayer.input_layernorm.weight"]); f=rms(feat,sd["midlayer.hidden_norm.weight"]); x=torch.cat([e,f])
    v=x@sd["midlayer.self_attn.v_proj.weight"].T; nH,nKV,hd=16,8,128;g=nH//nKV;ctx=torch.zeros(nH*hd)
    for qh in range(nH): ctx[qh*hd:(qh+1)*hd]=v[(qh//g)*hd:(qh//g+1)*hd]
    r=feat+ctx@sd["midlayer.self_attn.o_proj.weight"].T; x2=rms(r,sd["midlayer.post_attention_layernorm.weight"])
    r=r+(F.silu(x2@sd["midlayer.mlp.gate_proj.weight"].T)*(x2@sd["midlayer.mlp.up_proj.weight"].T))@sd["midlayer.mlp.down_proj.weight"].T
    return rms(r,sd["norm.weight"])@sd["lm_head.weight"].T
# CHAT format: let the model GENERATE an assistant turn (in-distribution), then score α over it.
msgs=[{"role":"user","content":"Explain how a transformer neural network works, in a few sentences."}]
_txt=tok.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False); pid=tok(_txt,return_tensors="pt").input_ids
with torch.no_grad(): gen=model.generate(pid,max_new_tokens=40,do_sample=False)
ids=gen; plen=pid.shape[1]
with torch.no_grad(): out=model(ids,output_hidden_states=True)
hs=out.hidden_states; logits=out.logits[0]; T=ids.shape[1]
def alpha(aux, rng):
    tot=0;n=0
    for t in rng:
        if t>=T-1: break
        h3=torch.cat([hs[a][0,t] for a in aux]); dl=step0(h3,ids[0,t].item()); q=F.softmax(dl,-1)
        p=F.softmax(logits[t],-1); qt=torch.zeros_like(p); qt[torch.arange(32000)+d2t]=q
        tot+=torch.minimum(p,qt).sum().item(); n+=1
    return tot/max(n,1)
# score over the GENERATED assistant tokens (in-distribution for the head)
for aux in [[3,15,27],[2,14,25],[5,15,27],[1,12,26]]:
    print("chat-gen aux",aux,"alpha=%.3f"%alpha(aux,range(plen,T-1)))
