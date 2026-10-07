import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel
torch.manual_seed(0)
cfg = AutoConfig.from_pretrained('/Users/francistownsend-merino/models/embeddinggemma-2')
ac = cfg.audio_config
small = type(ac)(**{**ac.to_dict(), 'hidden_size':64, 'num_attention_heads':2, 'num_hidden_layers':2, 'output_proj_dims':48})
m = Gemma4AudioModel(small).float().eval()
cap={}
orig=m._convert_4d_mask_to_blocked_5d
def hook5(mask4):
    cap['m4']=mask4.clone(); return orig(mask4)
m._convert_4d_mask_to_blocked_5d=hook5
feats=torch.randn(1,200,128); mask=torch.ones(1,200,dtype=torch.bool); mask[:,151:]=False; feats[~mask]=0
outs={}
for impl in ['sdpa','eager','flex_attention']:
    m.config._attn_implementation=impl
    try:
        outs[impl]=m(feats,mask).last_hidden_state; print(impl,'4d mask dtype',cap['m4'].dtype, 'unique', torch.unique(cap['m4'])[:4].tolist())
    except Exception as e: print(impl,'ERR',type(e).__name__, str(e)[:150])
v=torch.zeros(50,dtype=torch.bool); v[:38]=True
print('sdpa vs eager maxdiff on valid', (outs['sdpa'][0][v]-outs['eager'][0][v]).abs().max().item())
