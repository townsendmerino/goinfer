import torch, numpy as np, time, math
from safetensors import safe_open
from transformers import AutoConfig, AutoFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4ClippableLinear
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder
D='/Users/francistownsend-merino/models/embeddinggemma-2'
t0=time.time()
cfg=AutoConfig.from_pretrained(D); ac=cfg.audio_config
m=Gemma4AudioModel(ac).float().eval(); emb=EmbeddingGemma2MultimodalEmbedder(ac,cfg.text_config).float().eval()
sd={}; esd={}
with safe_open(D+'/model.safetensors','pt') as f:
    for k in f.keys():
        if k.startswith('audio_tower.'): sd[k[len('audio_tower.'):]]=f.get_tensor(k).float()
        if k.startswith('embed_audio.'): esd[k[len('embed_audio.'):]]=f.get_tensor(k).float()
r=m.load_state_dict(sd,strict=False); print('audio missing',r.missing_keys,'unexpected',r.unexpected_keys)
r=emb.load_state_dict(esd,strict=False); print('embed missing',r.missing_keys,'unexpected',r.unexpected_keys)
print('n params audio', sum(p.numel() for p in m.parameters()), 'attn impl', m.config._attn_implementation, 'load s', round(time.time()-t0,1))
fe=AutoFeatureExtractor.from_pretrained(D)
sr=16000; t=np.arange(int(2.37*sr))/sr
rng=np.random.default_rng(0)
x=(0.3*np.sin(2*np.pi*(200+600*t)*t)+0.02*rng.standard_normal(len(t))).astype(np.float32)
o=fe([x],return_tensors='pt'); feats,fm=o['input_features'],o['input_features_mask']
print('mel', tuple(feats.shape), 'valid', fm.sum().item())
# count clamp activations
stats={}
def mk(name):
    def h(mod, inp, out):
        x=inp[0]; lin=mod.linear(torch.clamp(x,mod.input_min,mod.input_max))
        stats[name]=(int(((x<mod.input_min)|(x>mod.input_max)).sum()), int(((lin<mod.output_min)|(lin>mod.output_max)).sum()), x.numel(), lin.numel())
    return h
hs=[mod.register_forward_hook(mk(n)) for n,mod in m.named_modules() if isinstance(mod,Gemma4ClippableLinear)]
with torch.no_grad(): out=m(feats,fm)
for h in hs: h.remove()
act=[(n,s) for n,s in stats.items() if s[0] or s[1]]
print('clippable linears', len(stats), 'with any clamp active', len(act)); [print('  ',n,s) for n,s in act[:12]]
with torch.no_grad():
    base=out.last_hidden_state; v=out.attention_mask[0]
    pooled=emb(base)
    print('tower out', tuple(base.shape), 'valid tokens', v.sum().item(), 'embedder out', tuple(pooled.shape), 'emb out rms', pooled[0][v].pow(2).mean().sqrt().item())
    # ablations
    def run_with(fn):
        fn(True); r=m(feats,fm).last_hidden_state; fn(False); return r
    saved={n:(mod.input_min.clone(),mod.input_max.clone(),mod.output_min.clone(),mod.output_max.clone()) for n,mod in m.named_modules() if isinstance(mod,Gemma4ClippableLinear)}
    def noclip(on):
        for n,mod in m.named_modules():
            if isinstance(mod,Gemma4ClippableLinear):
                if on: mod.input_min.fill_(-math.inf); mod.input_max.fill_(math.inf); mod.output_min.fill_(-math.inf); mod.output_max.fill_(math.inf)
                else: a,b,c,d=saved[n]; mod.input_min.copy_(a); mod.input_max.copy_(b); mod.output_min.copy_(c); mod.output_max.copy_(d)
    def cos(a,b): a=a[0][v].flatten(); b=b[0][v].flatten(); return (a@b/(a.norm()*b.norm())).item()
    r=run_with(noclip); print('no-clip cosine vs ref', cos(r,base), 'maxdiff', (r-base)[0][v].abs().max().item())
    m.config._attn_implementation='eager'; r=m(feats,fm).last_hidden_state; m.config._attn_implementation='sdpa'
    print('eager cosine vs sdpa ref', cos(r,base))
    # bf16 run
    mb=Gemma4AudioModel(ac).eval(); mb.load_state_dict({k:v_.bfloat16() for k,v_ in sd.items()},strict=False); mb=mb.bfloat16()
    rb=mb(feats.bfloat16(),fm).last_hidden_state.float(); print('bf16 tower cosine vs f32', cos(rb,base))
    # per-layer hidden rms
    hsx=[]
    hh=[l.register_forward_hook(lambda mod,i,o_: hsx.append(o_.detach())) for l in m.layers]
    m(feats,fm); [h.remove() for h in hh]
    print('per-layer out rms', [round(h_[0][v].pow(2).mean().sqrt().item(),3) for h_ in hsx])
print('total s', round(time.time()-t0,1))
