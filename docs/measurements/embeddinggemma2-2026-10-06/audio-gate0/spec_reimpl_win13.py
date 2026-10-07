# Independent re-implementation of the Gate-0 spec (direct sliding-window attention, no blocking),
# compared against HF Gemma4AudioModel on the real audio-tower weights. f64 throughout.
import torch, numpy as np, math
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2MultimodalEmbedder
D='/Users/francistownsend-merino/models/embeddinggemma-2'
cfg=AutoConfig.from_pretrained(D); ac=cfg.audio_config
W={}
with safe_open(D+'/model.safetensors','pt') as f:
    for k in f.keys():
        if 'audio' in k: W[k]=f.get_tensor(k).double()
A=lambda k: W['audio_tower.'+k]
def rms(x,w=None,eps=1e-6):
    y=x*torch.pow(x.pow(2).mean(-1,keepdim=True)+eps,-0.5)
    return y*w if w is not None else y
def clin(x,p):
    x=torch.clamp(x,A(p+'.input_min'),A(p+'.input_max')); y=x@A(p+'.linear.weight').T
    return torch.clamp(y,A(p+'.output_min'),A(p+'.output_max'))
def spec(feat, nvalid):
    x=feat[:nvalid].clone()            # [T,128] valid mel frames only
    h=x[None,None]                    # [1,1,T,F]
    for i in range(2):
        h=F.conv2d(h,A(f'subsample_conv_projection.layer{i}.conv.weight'),stride=2,padding=1)
        hp=h.permute(0,2,3,1); mu=hp.mean(-1,keepdim=True); var=((hp-mu)**2).mean(-1,keepdim=True)
        hp=(hp-mu)/torch.sqrt(var+1e-6)*A(f'subsample_conv_projection.layer{i}.norm.weight')
        h=F.relu(hp).permute(0,3,1,2)
    T=h.shape[2]; h=h[0].permute(1,2,0).reshape(T,-1)   # [T, F*C], index f*32+c
    h=h@A('subsample_conv_projection.input_proj_linear.weight').T
    H,Dh=8,128
    inv=torch.exp(torch.arange(512,dtype=torch.float64)*-(math.log(10000.0)/511))
    PE=torch.stack([torch.cat([torch.sin(d*inv),torch.cos(d*inv)]) for d in range(13)])  # PE[d], d=0..12
    def ffn(x,p):
        r=x; y=rms(x,A(p+'.pre_layer_norm.weight')); y=clin(y,p+'.ffw_layer_1'); y=F.silu(y); y=clin(y,p+'.ffw_layer_2')
        y=rms(y,A(p+'.post_layer_norm.weight')); return r+0.5*y
    for L in range(12):
        p=f'layers.{L}.'
        h=ffn(h,p+'feed_forward1'); res=h
        y=rms(h,A(p+'norm_pre_attn.weight'))
        q=clin(y,p+'self_attn.q_proj').view(T,H,Dh); k=clin(y,p+'self_attn.k_proj').view(T,H,Dh); v=clin(y,p+'self_attn.v_proj').view(T,H,Dh)
        q=q*(Dh**-0.5/math.log(2))*F.softplus(A(p+'self_attn.per_dim_scale')); k=k*(math.log(1+math.e)/math.log(2))
        R=(PE@A(p+'self_attn.relative_k_proj.weight').T).view(13,H,Dh)
        out=torch.zeros(T,H,Dh,dtype=torch.float64)
        for t in range(T):
            ks=list(range(max(0,t-12),t+1)); d=torch.tensor([t-j for j in ks])
            lg=torch.einsum('hd,khd->hk',q[t],k[ks])+torch.einsum('hd,khd->hk',q[t],R[d])
            lg=50*torch.tanh(lg/50); a=torch.softmax(lg,-1); out[t]=torch.einsum('hk,khd->hd',a,v[ks])
        y=clin(out.reshape(T,-1),p+'self_attn.post'); y=rms(y,A(p+'norm_post_attn.weight')); h=res+y
        r=h; y=rms(h,A(p+'lconv1d.pre_layer_norm.weight')); y=clin(y,p+'lconv1d.linear_start'); y=y[:,:1024]*torch.sigmoid(y[:,1024:])
        y=F.conv1d(F.pad(y.T[None],(4,0)),A(p+'lconv1d.depthwise_conv1d.weight'),groups=1024)[0].T
        y=rms(y,A(p+'lconv1d.conv_norm.weight')); y=F.silu(y); y=clin(y,p+'lconv1d.linear_end'); h=r+y
        h=ffn(h,p+'feed_forward2'); h=rms(h,A(p+'norm_out.weight'))
    h=h@A('output_proj.weight').T+A('output_proj.bias')
    e=rms(h)@W['embed_audio.embedding_projection.weight'].T
    return h,e
m=Gemma4AudioModel(ac).float().eval(); m.load_state_dict({k[12:]:v.float() for k,v in W.items() if k.startswith('audio_tower.')})
emb=EmbeddingGemma2MultimodalEmbedder(ac,cfg.text_config).float().eval(); emb.load_state_dict({"embedding_projection.weight":W["embed_audio.embedding_projection.weight"].float()})
fe=AutoFeatureExtractor.from_pretrained(D)
sr=16000; rng=np.random.default_rng(0)
for secs in [2.37]:
    t=np.arange(int(secs*sr))/sr
    x=(0.3*np.sin(2*np.pi*(200+600*t)*t)+0.02*rng.standard_normal(len(t))).astype(np.float32)
    o=fe([x],return_tensors='pt'); feats=o['input_features'].double(); fm=o['input_features_mask']
    with torch.no_grad():
        ho=m(feats.float(),fm); vm=ho.attention_mask[0]; he=emb(ho.last_hidden_state)[0][vm].double(); hh=ho.last_hidden_state[0][vm].double()
        sh,se=spec(feats[0], int(fm.sum()))
    print(f'{secs}s mel valid {int(fm.sum())} tokens HF {int(vm.sum())} spec {sh.shape[0]}  tower maxabs diff {(sh-hh).abs().max().item():.3e}  embed maxabs diff {(se-he).abs().max().item():.3e}  (embed max |x| {he.abs().max().item():.2f})')
