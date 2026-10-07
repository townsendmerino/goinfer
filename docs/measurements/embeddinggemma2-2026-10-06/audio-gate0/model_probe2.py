import torch, numpy as np, math
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel
import torch.nn.functional as F
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained('/Users/francistownsend-merino/models/embeddinggemma-2')
ac = cfg.audio_config
small = type(ac)(**{**ac.to_dict(), 'hidden_size':64, 'num_attention_heads':2, 'num_hidden_layers':2, 'output_proj_dims':48})
for impl in ['sdpa','eager']:
    m = Gemma4AudioModel(small).float().eval()
    m.config._attn_implementation=impl
    with torch.no_grad():
        for n,p in m.named_parameters():
            if 'norm' in n or 'per_dim_scale' in n: p.copy_(torch.randn_like(p)*0.5+1.0)
    cap={}
    orig=m._convert_4d_mask_to_blocked_5d
    def hook5(mask4, orig=orig):
        cap['m4']=mask4.clone(); r=orig(mask4); cap['m5']=r.clone(); return r
    m._convert_4d_mask_to_blocked_5d=hook5
    T=200
    feats=torch.randn(1,T,128)
    for nvalid in [T, 151]:
        cap.clear()
        mask=torch.zeros(1,T,dtype=torch.bool); mask[:, :nvalid]=True
        f=feats.clone(); f[~mask]=0
        out=m(f, mask)
        m4=cap['m4'][0,0]; S=m4.shape[0]
        dists=sorted(set(q-k for q in range(S) for k in range(S) if m4[q,k]))
        print(impl,'nvalid',nvalid,'S',S,'valid tok',out.attention_mask.sum().item(),'allowed dists',dists, 'q=30 keys',[k for k in range(S) if m4[30,k]])
        print('  invalid query rows have any True:', m4[out.attention_mask[0].logical_not()].any().item(), ' keys of invalid region ever allowed for valid q:', m4[out.attention_mask[0]][:, ~out.attention_mask[0]].any().item())
    cap.clear()
    o_none=m(feats, None); print(impl,'mask=None -> 4d mask captured?', 'm4' in cap, cap['m4'].shape if 'm4' in cap else None)
    if 'm4' in cap:
        m4=cap['m4'][0,0]; S=m4.shape[0]; print('   dists', sorted(set(q-k for q in range(S) for k in range(S) if m4[q,k])))
    o_all=m(feats, torch.ones(1,T,dtype=torch.bool)); print('   mask=None vs all-ones maxdiff', (o_none.last_hidden_state-o_all.last_hidden_state).abs().max().item())
# pos embed
pe=m.rel_pos_enc(torch.zeros(1,1,64)); its=m.rel_pos_enc.inv_timescales[0,0]
print('pos_embed shape', tuple(pe.shape), 'row r == pos (12-r)?', all(torch.allclose(pe[0,r], torch.cat([torch.sin((12-r)*its), torch.cos((12-r)*its)])) for r in range(13)))
print('inv_timescales[:3]', its[:3].tolist(), 'expected', [math.exp(-i*math.log(10000)/31) for i in range(3)])
# rel shift: verify out[i,j] = bd[i, j-i] for 0<=j-i<=12 else 0
att=m.layers[0].self_attn
x=torch.randn(1,2,3,12,13)
y=att._rel_shift(x)
ok=True
for i in range(12):
    for j in range(24):
        c=j-i
        exp = x[...,i,c] if 0<=c<=12 else torch.zeros_like(x[...,i,0])
        ok &= torch.equal(y[...,i,j], exp)
print('rel_shift formula out[i,j]=bd[i,j-i] (0<=j-i<=12) else 0 :', ok)
print('q_scale', att.q_scale, '= 128^-0.5/ln2 ->', (128**-0.5)/math.log(2), 'k_scale', att.k_scale, 'context_size', att.context_size, 'rel ctx', m.rel_pos_enc.context_size)
