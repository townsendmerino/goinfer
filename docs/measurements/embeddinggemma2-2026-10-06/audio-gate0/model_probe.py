import torch, numpy as np, math
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained('/Users/francistownsend-merino/models/embeddinggemma-2')
ac = cfg.audio_config
print('audio cfg class', type(ac).__name__, 'attn impl', getattr(ac,'_attn_implementation',None))
small = type(ac)(**{**ac.to_dict(), 'hidden_size':64, 'num_attention_heads':2, 'num_hidden_layers':2, 'output_proj_dims':48})
small.subsampling_conv_channels=[128,32]
m = Gemma4AudioModel(small).float().eval()
print('attn impl small', m.config._attn_implementation)
# randomize per_dim_scale and norms so nothing is identity
with torch.no_grad():
    for n,p in m.named_parameters():
        if 'norm' in n or 'per_dim_scale' in n: p.copy_(torch.randn_like(p)*0.5+1.0)
# capture masks
cap={}
orig=m._convert_4d_mask_to_blocked_5d
def hook5(mask4):
    cap['m4']=mask4.clone(); r=orig(mask4); cap['m5']=r.clone(); return r
m._convert_4d_mask_to_blocked_5d=hook5
T=40  # mel frames -> 20 -> 10 subsampled
feats=torch.randn(1,T,128)
for nvalid in [T, 27]:
    cap.clear()
    mask=torch.zeros(1,T,dtype=torch.bool); mask[:, :nvalid]=True
    f=feats.clone(); f[~mask]=0
    out=m(f, mask)
    print('nvalid',nvalid,'out mask sum', out.attention_mask.sum().item(), 'shape', out.last_hidden_state.shape, 'm4 is None?', 'm4' not in cap)
    if 'm4' in cap:
        m4=cap['m4'][0,0]; S=m4.shape[0]
        allowed=[(q,k) for q in range(S) for k in range(S) if m4[q,k]]
        dists=sorted(set(q-k for q,k in allowed)); print(' 4d mask dtype',cap['m4'].dtype,'S',S,'allowed dists', dists)
        print(' row q=S-1 allowed keys', [k for k in range(S) if m4[S-1,k]], ' invalid q row (last) any?', m4[-1].any().item())
        m5=cap['m5']; print(' 5d shape', tuple(m5.shape))
        # per block decode: q abs = b*12+i, key abs = b*12-12+j
        bad=0
        for b in range(m5.shape[2]):
            for i in range(12):
                for j in range(m5.shape[4]):
                    q=b*12+i; k=b*12-12+j
                    exp = (0<=k<S) and (q<S) and bool(m4[q,k]) if (0<=k<S and q<S) else False
                    if bool(m5[0,0,b,i,j])!=exp: bad+=1
        print(' 5d mismatches vs 4d decode', bad)
# padded vs truncated equivalence
nvalid=27
mask=torch.zeros(1,T,dtype=torch.bool); mask[:, :nvalid]=True
f=feats.clone(); f[~mask]=0
o1=m(f,mask); v=o1.attention_mask[0]
o2=m(f[:, :nvalid], mask[:, :nvalid])
print('valid tokens padded', v.sum().item(), 'trunc', o2.attention_mask.sum().item(), 'trunc len', o2.last_hidden_state.shape[1])
print('padded-vs-trunc maxdiff', (o1.last_hidden_state[0][v]-o2.last_hidden_state[0][o2.attention_mask[0]]).abs().max().item())
# garbage in invalid frames
f3=f.clone(); f3[~mask]=torch.randn_like(f3[~mask])*100
o3=m(f3,mask); print('garbage-in-invalid maxdiff', (o3.last_hidden_state[0][v]-o1.last_hidden_state[0][v]).abs().max().item())
# mask=None path
o4=m(f[:, :nvalid], None); print('mask=None vs all-valid mask maxdiff', (o4.last_hidden_state-o2.last_hidden_state).abs().max().item(), 'o4 mask', o4.attention_mask)
# rel shift check
att=m.layers[0].self_attn
pe=m.rel_pos_enc(torch.zeros(1,1,64))
print('pos_embed shape', tuple(pe.shape), 'row0 == pos 12?', torch.allclose(pe[0], torch.cat([torch.sin(12*m.rel_pos_enc.inv_timescales[0,0]), torch.cos(12*m.rel_pos_enc.inv_timescales[0,0])])))
x=torch.randn(1,1,2,3,13)  # B,H,nb,chunk(=3? no)
